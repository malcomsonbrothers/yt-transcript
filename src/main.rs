use std::env;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Deserialize;

const DEFAULT_MODEL_ID: &str = "nvidia/parakeet-tdt-0.6b-v3";
const LOCAL_TRANSCRIBE_SCRIPT: &str = include_str!("nemo_transcribe.py");

/// Together AI rejects direct uploads above 80 MB. Oversized audio is re-encoded
/// rather than refused; see `shrink_for_upload`.
const MAX_UPLOAD_BYTES: u64 = 80_000_000;
/// Parakeet resamples to 16 kHz mono, so a 16 kHz mono Opus stream discards only
/// what the model never receives. 32 kbps keeps Together's 4-hour ceiling under
/// 60 MB, so one re-encode is always enough.
const UPLOAD_SHRINK_SAMPLE_RATE: &str = "16000";
const UPLOAD_SHRINK_BITRATE: &str = "32k";

#[derive(Parser, Debug)]
#[command(
    name = "yt-transcript",
    version,
    about = "Transcribe YouTube audio with Together AI by default, or use local runtimes with --local"
)]
struct Cli {
    /// YouTube video URL to download audio from.
    url: Option<String>,

    /// Model identifier (prefer Hugging Face IDs from `yt-transcript models list`).
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model: String,

    /// Directory where audio and transcript outputs are written.
    #[arg(long, default_value = ".")]
    output_dir: PathBuf,

    /// Explicit output path for the transcript text file.
    #[arg(long)]
    transcript_path: Option<PathBuf>,

    /// Path to yt-dlp executable.
    #[arg(long, default_value = "yt-dlp")]
    yt_dlp_path: String,

    /// Path to ffmpeg executable.
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg_path: String,

    /// Path to uv executable used for local model runtime.
    #[arg(long, default_value = "uv")]
    uv_path: String,

    /// Python version for `uv run --python`.
    #[arg(long, default_value = "3.12")]
    python_version: String,

    /// Optional Hugging Face token for gated model downloads.
    #[arg(long, env = "HF_TOKEN")]
    hf_token: Option<String>,

    /// Together AI API key for hosted transcription.
    #[arg(long, env = "TOGETHER_API_KEY")]
    together_api_key: Option<String>,

    /// Use local runtimes only; otherwise Together AI is required.
    #[arg(long)]
    local: bool,

    /// Request speaker diarisation from Together AI.
    #[arg(long)]
    diarize: bool,

    /// Fix the requested minimum and maximum speaker count.
    #[arg(long)]
    speakers: Option<u8>,

    /// Include segment timestamps in the transcript.
    #[arg(long)]
    timestamps: bool,

    /// Device selection for local inference.
    #[arg(long, value_enum, default_value_t = DeviceMode::Auto)]
    device: DeviceMode,

    /// max_new_tokens used for Canary generation.
    #[arg(long, default_value_t = 256)]
    canary_max_new_tokens: u16,

    /// Disable yt-dlp download progress output.
    #[arg(long)]
    no_download_progress: bool,

    /// Disable local transcription progress output.
    #[arg(long)]
    no_transcribe_progress: bool,

    /// Chunk size in seconds used for MLX chunked transcription progress updates.
    #[arg(long, default_value_t = 120.0)]
    mlx_chunk_seconds: f32,

    /// Print subprocess commands before running.
    #[arg(long)]
    print_command: bool,

    /// Remove downloaded audio after transcript is written.
    #[arg(long)]
    delete_audio: bool,

    /// Also download and keep the full video next to the audio.
    #[arg(long)]
    keep_video: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Model-related commands.
    Models {
        #[command(subcommand)]
        command: Option<ModelsCommands>,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsCommands {
    /// List all supported model IDs and aliases.
    List,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptionBackend {
    TogetherCloud,
    Local,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelRuntime {
    TogetherCloud,
    ParakeetMlx,
    ParakeetNemo,
    CanaryNemo,
}

impl ModelRuntime {
    fn as_script_value(self) -> &'static str {
        match self {
            Self::TogetherCloud => "together_cloud",
            Self::ParakeetMlx => "parakeet_mlx",
            Self::ParakeetNemo => "parakeet_nemo",
            Self::CanaryNemo => "canary_nemo",
        }
    }

    fn short_name(self) -> &'static str {
        match self {
            Self::TogetherCloud => "together",
            Self::ParakeetMlx => "parakeet-mlx",
            Self::ParakeetNemo => "parakeet-nemo",
            Self::CanaryNemo => "canary-nemo",
        }
    }

    fn dependency_packages(self) -> &'static [&'static str] {
        match self {
            Self::TogetherCloud => &[],
            Self::ParakeetMlx => &["parakeet-mlx"],
            Self::ParakeetNemo | Self::CanaryNemo => &["torch", "nemo_toolkit[asr]"],
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::TogetherCloud => "Together AI hosted inference (no local GPU or Python)",
            Self::ParakeetMlx => "local MLX runtime via `uv run --with parakeet-mlx`",
            Self::ParakeetNemo | Self::CanaryNemo => {
                "local NeMo runtime via `uv run --with torch --with nemo_toolkit[asr]`"
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ModelProfile {
    id: &'static str,
    display_name: &'static str,
    aliases: &'static [&'static str],
    notes: &'static str,
    yt_dlp_format: &'static str,
    output_format: &'static str,
    sample_rate_hz: u32,
    channels: u8,
    runtime: ModelRuntime,
    mlx_model_id: Option<&'static str>,
    together_model_id: Option<&'static str>,
}

#[derive(Debug)]
struct VideoMeta {
    id: String,
    safe_title: String,
}

#[derive(Debug)]
struct DownloadConfig<'a> {
    output_dir: &'a Path,
    yt_dlp_path: &'a str,
    ffmpeg_path: &'a str,
    cloud_mode: bool,
    print_command: bool,
    no_download_progress: bool,
}

#[derive(Debug)]
struct LocalTranscriptionConfig<'a> {
    uv_path: &'a str,
    ffmpeg_path: &'a str,
    together_api_key: Option<&'a str>,
    force_local: bool,
    diarize: bool,
    speakers: Option<u8>,
    timestamps: bool,
    python_version: &'a str,
    hf_token: Option<&'a str>,
    device: DeviceMode,
    canary_max_new_tokens: u16,
    transcribe_progress: bool,
    mlx_chunk_seconds: f32,
    print_command: bool,
}

#[derive(Debug, Deserialize)]
struct LocalTranscriptionResult {
    transcript: String,
    device: String,
    model_id: String,
    runtime: String,
    #[serde(default)]
    audio_duration_seconds: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct TogetherTranscriptionResponse {
    duration: f64,
    text: String,
    segments: Vec<TogetherSegment>,
    #[serde(default)]
    speaker_segments: Vec<TogetherSpeakerSegment>,
}

#[derive(Debug, Deserialize)]
struct TogetherSegment {
    text: String,
    start: f64,
    #[serde(alias = "speaker_id")]
    speaker: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TogetherSpeakerSegment {
    text: String,
    start: f64,
    speaker_id: String,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceMode {
    Auto,
    Mps,
    Cuda,
    Cpu,
}

impl DeviceMode {
    fn as_arg(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Mps => "mps",
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }
}

/// Fallback chain for sites that only publish muxed (video+audio) streams.
///
/// TikTok labels every muxed format as carrying AAC audio, but its H.265
/// (`bytevc1`) files often have no audio track at all, and plain `best`
/// picks exactly those. H.264 streams reliably carry the audio, so they are
/// tried first; plain `best` stays last for sites without H.264.
macro_rules! muxed_fallback {
    () => {
        "best[vcodec^=h264]/best[vcodec^=avc1]/best"
    };
}

const MODELS: [ModelProfile; 2] = [
    ModelProfile {
        id: "nvidia/parakeet-tdt-0.6b-v3",
        display_name: "NVIDIA Parakeet TDT 0.6B v3",
        aliases: &["parakeet", "parakeet-v3"],
        notes: "Default. Optimized for high-throughput multilingual transcription.",
        yt_dlp_format: concat!(
            "bestaudio[acodec*=opus]/bestaudio[abr>=128]/bestaudio/",
            muxed_fallback!()
        ),
        output_format: "wav",
        sample_rate_hz: 16_000,
        channels: 1,
        runtime: ModelRuntime::ParakeetNemo,
        mlx_model_id: Some("mlx-community/parakeet-tdt-0.6b-v3"),
        together_model_id: Some("nvidia/parakeet-tdt-0.6b-v3"),
    },
    ModelProfile {
        id: "nvidia/canary-qwen-2.5b",
        display_name: "NVIDIA Canary Qwen 2.5B",
        aliases: &["canary", "canary-qwen-2.5b"],
        notes: "Higher-accuracy English model.",
        yt_dlp_format: concat!(
            "bestaudio[asr>=44100]/bestaudio[abr>=160]/bestaudio/",
            muxed_fallback!()
        ),
        output_format: "wav",
        sample_rate_hz: 16_000,
        channels: 1,
        runtime: ModelRuntime::CanaryNemo,
        mlx_model_id: None,
        together_model_id: None,
    },
];

fn main() -> Result<()> {
    let total_started_at = Instant::now();
    let cli = Cli::parse();

    if let Some(command) = cli.command {
        return handle_command(command);
    }

    let together_api_key = cli
        .together_api_key
        .as_deref()
        .filter(|key| !key.trim().is_empty());
    let model = resolve_model(&cli.model).ok_or_else(|| {
        anyhow!(
            "unknown model `{}`; run `yt-transcript models list`",
            cli.model
        )
    })?;
    validate_transcription_options(&cli, model, together_api_key)?;

    let url = cli
        .url
        .as_deref()
        .context("a URL is required unless using a subcommand")?;
    let backend = transcription_backend(cli.local);
    let cloud_mode = backend == TranscriptionBackend::TogetherCloud;

    fs::create_dir_all(&cli.output_dir).with_context(|| {
        format!(
            "failed to create output directory `{}`",
            cli.output_dir.display()
        )
    })?;

    stage(&format!("resolving video metadata for {url}"));
    let metadata_started_at = Instant::now();
    let video_meta = fetch_video_metadata(url, &cli.yt_dlp_path, cli.print_command)?;
    let metadata_duration = metadata_started_at.elapsed();
    stage(&format!(
        "video metadata resolved in {}",
        format_duration(metadata_duration)
    ));

    stage(&format!("downloading audio for model {}", model.id));
    let download_started_at = Instant::now();
    let download_config = DownloadConfig {
        output_dir: &cli.output_dir,
        yt_dlp_path: &cli.yt_dlp_path,
        ffmpeg_path: &cli.ffmpeg_path,
        cloud_mode,
        print_command: cli.print_command,
        no_download_progress: cli.no_download_progress,
    };
    let audio_path = download_audio(url, model, &video_meta, &download_config)?;
    let download_duration = download_started_at.elapsed();
    stage(&format!(
        "audio downloaded in {}",
        format_duration(download_duration)
    ));

    let video_path = if cli.keep_video {
        stage("downloading video to keep");
        let video_started_at = Instant::now();
        let path = download_video(url, &video_meta, &download_config)?;
        let video_bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        stage(&format!(
            "video downloaded in {} ({})",
            format_duration(video_started_at.elapsed()),
            format_megabytes(video_bytes)
        ));
        Some(path)
    } else {
        None
    };

    stage(&format!(
        "transcribing with {} ({})",
        model.display_name, model.id
    ));
    let transcribe_started_at = Instant::now();
    let local_transcription = transcribe_audio_local(
        &audio_path,
        model,
        &LocalTranscriptionConfig {
            uv_path: &cli.uv_path,
            ffmpeg_path: &cli.ffmpeg_path,
            together_api_key,
            force_local: cli.local,
            diarize: cli.diarize,
            speakers: cli.speakers,
            timestamps: cli.timestamps,
            python_version: &cli.python_version,
            hf_token: cli.hf_token.as_deref(),
            device: cli.device,
            canary_max_new_tokens: cli.canary_max_new_tokens,
            transcribe_progress: !cli.no_transcribe_progress,
            mlx_chunk_seconds: cli.mlx_chunk_seconds,
            print_command: cli.print_command,
        },
    )?;
    let transcribe_duration = transcribe_started_at.elapsed();
    stage(&format!(
        "transcription finished in {}",
        format_duration(transcribe_duration)
    ));

    if local_transcription.model_id != model.id {
        bail!(
            "transcriber returned model `{}` but `{}` was requested",
            local_transcription.model_id,
            model.id
        );
    }

    let transcript_path = build_transcript_path(&cli, &video_meta);
    if let Some(parent) = transcript_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create `{}`", parent.display()))?;
    }

    let audio_seconds = if cli.local {
        try_wav_duration_seconds(&audio_path)
    } else {
        local_transcription.audio_duration_seconds
    };

    let write_started_at = Instant::now();
    fs::write(&transcript_path, local_transcription.transcript).with_context(|| {
        format!(
            "failed to write transcript to `{}`",
            transcript_path.display()
        )
    })?;
    let write_duration = write_started_at.elapsed();

    if cli.delete_audio {
        remove_audio_file(&audio_path)?;
    }

    let total_duration = total_started_at.elapsed();
    stage("done");
    println!("audio_file={}", audio_path.display());
    if let Some(video_path) = &video_path {
        println!("video_file={}", video_path.display());
    }
    println!("transcript_file={}", transcript_path.display());
    println!("device={}", local_transcription.device);
    println!("runtime={}", local_transcription.runtime);
    println!("timing_metadata={}", format_duration(metadata_duration));
    println!("timing_download={}", format_duration(download_duration));
    println!(
        "timing_transcription={}",
        format_duration(transcribe_duration)
    );
    println!("timing_write={}", format_duration(write_duration));
    println!("timing_total={}", format_duration(total_duration));
    if let Some(audio_seconds) = audio_seconds {
        println!("audio_duration={audio_seconds:.2}s");
        if transcribe_duration.as_secs_f64() > 0.0 {
            println!(
                "transcription_speed={:.2}x_realtime",
                audio_seconds / transcribe_duration.as_secs_f64()
            );
        }
    }

    Ok(())
}

fn validate_transcription_options(
    cli: &Cli,
    model: &ModelProfile,
    together_api_key: Option<&str>,
) -> Result<()> {
    let diarisation_requested = cli.diarize || cli.speakers.is_some();
    if diarisation_requested && cli.local {
        bail!(
            "diarisation is only available with Together AI; remove `--local` and set TOGETHER_API_KEY"
        );
    }
    if diarisation_requested && model.together_model_id.is_none() {
        bail!(
            "diarisation is unavailable for model `{}` because it is not hosted by Together AI; choose a hosted model",
            model.id
        );
    }
    if diarisation_requested && together_api_key.is_none() {
        bail!(
            "diarisation requires the Together cloud backend; set TOGETHER_API_KEY or remove the diarisation options and pass `--local`"
        );
    }
    if !cli.local && model.together_model_id.is_none() {
        bail!(
            "model `{}` is local-only; pass `--local` to use it",
            model.id
        );
    }
    if !cli.local && together_api_key.is_none() {
        bail!(
            "Together AI cloud transcription requires TOGETHER_API_KEY; set it or pass `--local` to transcribe locally"
        );
    }
    Ok(())
}

fn remove_audio_file(path: &Path) -> Result<()> {
    fs::remove_file(path)
        .with_context(|| format!("failed to delete audio file `{}`", path.display()))
}

fn handle_command(command: Commands) -> Result<()> {
    match command {
        Commands::Models { command } => {
            let selected = command.unwrap_or(ModelsCommands::List);
            match selected {
                ModelsCommands::List => {
                    print_models();
                    Ok(())
                }
            }
        }
    }
}

fn resolve_model(input: &str) -> Option<&'static ModelProfile> {
    let normalized = input.trim().to_ascii_lowercase();

    MODELS.iter().find(|model| {
        model.id.eq_ignore_ascii_case(&normalized)
            || model
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(&normalized))
    })
}

fn print_models() {
    for model in MODELS {
        let default_suffix = if model.id == DEFAULT_MODEL_ID {
            " (default)"
        } else {
            ""
        };

        println!("{}{}", model.id, default_suffix);
        println!("  name: {}", model.display_name);
        println!("  notes: {}", model.notes);
        println!("  aliases: {}", model.aliases.join(", "));
        println!("  runtime: {}", runtime_summary(&model));
        println!();
    }
}

fn runtime_summary(model: &ModelProfile) -> String {
    let local_summary = if cfg!(target_os = "macos") && model.mlx_model_id.is_some() {
        format!(
            "{} -> {} fallback",
            ModelRuntime::ParakeetMlx.description(),
            ModelRuntime::ParakeetNemo.short_name(),
        )
    } else {
        model.runtime.description().to_string()
    };

    if model.together_model_id.is_some() {
        format!(
            "default: {} (requires TOGETHER_API_KEY); --local: {local_summary}",
            ModelRuntime::TogetherCloud.description()
        )
    } else {
        format!("local-only (requires --local): {local_summary}")
    }
}

fn fetch_video_metadata(url: &str, yt_dlp_path: &str, print_command: bool) -> Result<VideoMeta> {
    let mut command = Command::new(yt_dlp_path);
    command
        .arg("--no-warnings")
        .arg("--no-playlist")
        .arg("--skip-download")
        .arg("--print")
        .arg("%(id)s\t%(title)s")
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    if print_command {
        eprintln!("{}", render_command(&command));
    }

    let output = command
        .output()
        .with_context(|| format!("failed to execute `{}`", yt_dlp_path))?;

    if !output.status.success() {
        bail!(
            "yt-dlp metadata lookup failed with status {}",
            output.status
        );
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let line = text
        .lines()
        .find(|candidate| !candidate.trim().is_empty())
        .context("yt-dlp did not return video metadata")?;

    let (id, raw_title) = line
        .split_once('\t')
        .context("unexpected metadata response from yt-dlp")?;

    let safe_title = sanitize_filename(raw_title);
    let safe_title = if safe_title.is_empty() {
        "video".to_string()
    } else {
        safe_title
    };

    Ok(VideoMeta {
        id: id.to_string(),
        safe_title,
    })
}

/// Resolve the ffmpeg binary, failing early when an explicit path is wrong.
fn resolve_ffmpeg_location(ffmpeg_path: &str) -> Result<Option<PathBuf>> {
    let location = resolve_executable_path(ffmpeg_path);
    let arg_is_path = Path::new(ffmpeg_path).components().count() > 1;

    if arg_is_path && location.is_none() {
        bail!("ffmpeg path `{ffmpeg_path}` does not exist or is not executable");
    }

    Ok(location)
}

/// yt-dlp arguments for the file that gets transcribed.
///
/// Cloud mode uploads the source container as-is, so it only needs an audio
/// track. Some sites publish muxed streams only (Dailymotion, for example), so
/// the selector falls back to `best` and `--extract-audio` splits the audio
/// track out. `--audio-format best` keeps the source codec, which makes that
/// fallback a remux rather than a re-encode.
fn audio_download_args(
    model: &ModelProfile,
    cloud_mode: bool,
    ffmpeg_path: &str,
) -> Result<Vec<String>> {
    let mut args: Vec<String> = vec!["--no-playlist".to_string()];

    if cloud_mode {
        args.extend(
            [
                "--extract-audio",
                "--audio-format",
                "best",
                "-f",
                concat!("bestaudio/", muxed_fallback!()),
            ]
            .map(str::to_string),
        );
    } else {
        let postprocessor_args = format!(
            "ffmpeg:-ac {} -ar {} -sample_fmt s16",
            model.channels, model.sample_rate_hz
        );
        args.extend(
            [
                "--extract-audio",
                "--audio-format",
                model.output_format,
                "--audio-quality",
                "0",
                "--postprocessor-args",
                postprocessor_args.as_str(),
                "-f",
                model.yt_dlp_format,
            ]
            .map(str::to_string),
        );
    }

    if let Some(path) = resolve_ffmpeg_location(ffmpeg_path)? {
        args.push("--ffmpeg-location".to_string());
        args.push(path.display().to_string());
    }

    Ok(args)
}

/// yt-dlp arguments for the optional kept video.
///
/// `bestvideo*+bestaudio` prefers separate streams and falls back to a
/// single muxed one (H.264 first, see `muxed_fallback!`) on sites that only
/// publish that. Merging to MP4 keeps the
/// container predictable across sites.
fn video_download_args() -> [&'static str; 5] {
    [
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-f",
        concat!("bestvideo*+bestaudio/", muxed_fallback!()),
    ]
}

/// Download the full video so the source media survives next to the transcript.
fn download_video(url: &str, meta: &VideoMeta, config: &DownloadConfig<'_>) -> Result<PathBuf> {
    let base_name = format!("{}-{}", meta.safe_title, meta.id);
    let output_template = config.output_dir.join(format!("{base_name}-video.%(ext)s"));

    let mut command = Command::new(config.yt_dlp_path);
    for arg in video_download_args() {
        command.arg(arg);
    }

    command
        .arg("-o")
        .arg(output_template)
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if config.no_download_progress {
        command.arg("--no-progress");
    }

    if config.print_command {
        eprintln!("{}", render_command(&command));
    }

    let status = command
        .status()
        .with_context(|| format!("failed to execute `{}`", config.yt_dlp_path))?;

    if !status.success() {
        bail!("yt-dlp video download failed with status {status}");
    }

    find_downloaded_video(config.output_dir, &base_name)
}

/// Locate the kept video, preferring the MP4 that `--merge-output-format` asks
/// for and otherwise accepting whatever container yt-dlp produced.
fn find_downloaded_video(output_dir: &Path, base_name: &str) -> Result<PathBuf> {
    let preferred = output_dir.join(format!("{base_name}-video.mp4"));
    if preferred.is_file() {
        return Ok(preferred);
    }

    let stem = format!("{base_name}-video");
    let mut candidates: Vec<PathBuf> = fs::read_dir(output_dir)
        .with_context(|| format!("failed to read `{}`", output_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.file_stem().and_then(|value| value.to_str()) == Some(stem.as_str())
                && path
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| extension != "part" && extension != "ytdl")
        })
        .collect();
    candidates.sort();

    candidates
        .pop()
        .with_context(|| format!("yt-dlp did not produce a video file for `{base_name}`"))
}

fn download_audio(
    url: &str,
    model: &ModelProfile,
    meta: &VideoMeta,
    config: &DownloadConfig<'_>,
) -> Result<PathBuf> {
    let base_name = format!("{}-{}", meta.safe_title, meta.id);
    let output_template = config.output_dir.join(format!("{base_name}.%(ext)s"));
    let output_audio = config
        .output_dir
        .join(format!("{base_name}.{}", model.output_format));

    let mut command = Command::new(config.yt_dlp_path);
    for arg in audio_download_args(model, config.cloud_mode, config.ffmpeg_path)? {
        command.arg(arg);
    }

    let downloaded_path_report = config.cloud_mode.then(unique_downloaded_path_report);
    if let Some(report_path) = &downloaded_path_report {
        command
            .arg("--print-to-file")
            .arg("after_move:%(filepath)s")
            .arg(report_path)
            .arg("--no-simulate");
    }

    command
        .arg("-o")
        .arg(output_template)
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if config.no_download_progress {
        command.arg("--no-progress");
    }

    if config.print_command {
        eprintln!("{}", render_command(&command));
    }

    let status = command
        .status()
        .with_context(|| format!("failed to execute `{}`", config.yt_dlp_path))?;

    if !status.success() {
        if let Some(report_path) = &downloaded_path_report {
            let _ = fs::remove_file(report_path);
        }
        bail!("yt-dlp download failed with status {status}");
    }

    if let Some(report_path) = downloaded_path_report {
        return discover_downloaded_audio(&report_path, &base_name);
    }

    if !output_audio.exists() {
        bail!(
            "expected audio output `{}` was not produced",
            output_audio.display()
        );
    }

    Ok(output_audio)
}

fn discover_downloaded_audio(report_path: &Path, base_name: &str) -> Result<PathBuf> {
    const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a", "webm", "flac", "ogg", "opus", "aac"];

    let reported = fs::read_to_string(report_path).with_context(|| {
        format!(
            "yt-dlp did not report the downloaded audio path in `{}`",
            report_path.display()
        )
    });
    let _ = fs::remove_file(report_path);
    let reported = reported?;
    let path = reported
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .map(str::trim)
        .map(PathBuf::from)
        .with_context(|| format!("yt-dlp did not produce an audio file for `{base_name}`"))?;
    let supported = path.file_stem().and_then(|stem| stem.to_str()) == Some(base_name)
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                AUDIO_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str())
            });

    if !supported || !path.is_file() {
        bail!(
            "yt-dlp did not produce a supported audio file for `{base_name}` (reported `{}`)",
            path.display()
        );
    }

    Ok(path)
}

fn unique_downloaded_path_report() -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);

    env::temp_dir().join(format!("yt-transcript-download-{pid}-{nanos}.txt"))
}

fn resolve_executable_path(tool: &str) -> Option<PathBuf> {
    let raw = Path::new(tool);
    if raw.components().count() > 1 || raw.is_absolute() {
        if raw.is_file() {
            return Some(raw.to_path_buf());
        }
        return None;
    }

    let path = env::var_os("PATH")?;
    for entry in env::split_paths(&path) {
        let candidate = entry.join(tool);
        if candidate.is_file() {
            return Some(candidate);
        }

        #[cfg(windows)]
        {
            for ext in ["exe", "cmd", "bat"] {
                let candidate = entry.join(format!("{tool}.{ext}"));
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
    }

    None
}

fn transcribe_audio_local(
    audio_path: &Path,
    model: &ModelProfile,
    config: &LocalTranscriptionConfig<'_>,
) -> Result<LocalTranscriptionResult> {
    if !audio_path.exists() {
        bail!("audio file does not exist: `{}`", audio_path.display());
    }

    if transcription_backend(config.force_local) == TranscriptionBackend::TogetherCloud {
        let runtime = ModelRuntime::TogetherCloud;
        stage(&format!(
            "trying runtime {} ({})",
            runtime.short_name(),
            runtime.description()
        ));
        return run_together_runtime(audio_path, model, config).with_context(
            || "Together cloud transcription failed; pass `--local` to transcribe locally",
        );
    }

    let mut failures = Vec::new();
    for runtime in runtime_candidates(model) {
        stage(&format!(
            "trying runtime {} ({})",
            runtime.short_name(),
            runtime.description()
        ));

        match run_local_runtime(audio_path, model, runtime, config) {
            Ok(result) => return Ok(result),
            Err(error) => {
                failures.push(format!("{}: {error:#}", runtime.short_name()));
                stage(&format!(
                    "runtime {} failed, trying local fallback if available",
                    runtime.short_name()
                ));
            }
        }
    }

    bail!(
        "all local transcription runtimes failed: {}",
        failures.join(" | ")
    )
}

fn transcription_backend(local: bool) -> TranscriptionBackend {
    if local {
        TranscriptionBackend::Local
    } else {
        TranscriptionBackend::TogetherCloud
    }
}

fn runtime_candidates(model: &ModelProfile) -> Vec<ModelRuntime> {
    match model.runtime {
        ModelRuntime::TogetherCloud => Vec::new(),
        ModelRuntime::CanaryNemo => vec![ModelRuntime::CanaryNemo],
        ModelRuntime::ParakeetMlx => vec![ModelRuntime::ParakeetMlx],
        ModelRuntime::ParakeetNemo => {
            if cfg!(target_os = "macos") && model.mlx_model_id.is_some() {
                vec![ModelRuntime::ParakeetMlx, ModelRuntime::ParakeetNemo]
            } else {
                vec![ModelRuntime::ParakeetNemo]
            }
        }
    }
}

fn run_together_runtime(
    audio_path: &Path,
    model: &ModelProfile,
    config: &LocalTranscriptionConfig<'_>,
) -> Result<LocalTranscriptionResult> {
    const ENDPOINT: &str = "https://api.together.ai/v1/audio/transcriptions";

    let api_key = config
        .together_api_key
        .context("Together cloud runtime requires TOGETHER_API_KEY")?;
    let together_model_id = model
        .together_model_id
        .context("the selected model is not available on Together AI")?;
    let metadata = fs::metadata(audio_path)
        .with_context(|| format!("failed to inspect audio file `{}`", audio_path.display()))?;

    // Files over the limit are re-encoded to a speech-grade stream rather than
    // rejected. `_shrunk` keeps the temporary file alive for the whole request.
    let _shrunk = (metadata.len() > MAX_UPLOAD_BYTES)
        .then(|| shrink_for_upload(audio_path, metadata.len(), config))
        .transpose()?;
    let upload_path = _shrunk.as_ref().map_or(audio_path, |file| file.path());

    let file = fs::File::open(upload_path)
        .with_context(|| format!("failed to open audio file `{}`", upload_path.display()))?;
    let file_name = upload_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio")
        .to_string();
    let file_part = reqwest::blocking::multipart::Part::reader(file).file_name(file_name);
    let diarise = config.diarize || config.speakers.is_some();
    // Together rejects the request unless `model` arrives before the audio part,
    // and reports that ordering failure as HTTP 413. Keep `.text("model", ..)` first.
    let mut form = reqwest::blocking::multipart::Form::new()
        .text("model", together_model_id.to_string())
        .text("language", "en")
        .text("response_format", "verbose_json");
    if diarise {
        form = form.text("diarize", "true");
    }
    if let Some(speakers) = config.speakers {
        let speakers = speakers.to_string();
        form = form
            .text("min_speakers", speakers.clone())
            .text("max_speakers", speakers);
    }
    let form = form.part("file", file_part);

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30 * 60))
        .build()
        .context("failed to initialise Together AI HTTP client")?;
    let response = client
        .post(ENDPOINT)
        .bearer_auth(api_key)
        .multipart(form)
        .send()
        .context("Together AI transcription request failed")?;
    let status = response.status();
    let body = response
        .text()
        .context("failed to read Together AI response body")?;
    if !status.is_success() {
        let readable_body: String = body.chars().take(2_000).collect();
        bail!("Together AI returned HTTP {status}: {readable_body}");
    }

    let response: TogetherTranscriptionResponse =
        serde_json::from_str(&body).context("invalid Together AI transcription response JSON")?;
    let transcript = format_together_transcript(&response, config.timestamps, diarise);

    Ok(LocalTranscriptionResult {
        transcript,
        device: "together-cloud".into(),
        model_id: together_model_id.to_string(),
        runtime: "together_cloud".into(),
        audio_duration_seconds: Some(response.duration),
    })
}

/// A temporary file removed when dropped.
struct TempAudio {
    path: PathBuf,
}

impl TempAudio {
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempAudio {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Re-encode oversized audio to 16 kHz mono Opus so it fits Together's upload
/// limit. Lossy, but only below what the model can hear.
fn shrink_for_upload(
    audio_path: &Path,
    original_bytes: u64,
    config: &LocalTranscriptionConfig<'_>,
) -> Result<TempAudio> {
    let ffmpeg = resolve_executable_path(config.ffmpeg_path).ok_or_else(|| {
        anyhow!(
            "audio file `{}` is {} but Together AI accepts at most {}; \
             re-encoding needs ffmpeg on your PATH (or pass `--ffmpeg-path`)",
            audio_path.display(),
            format_megabytes(original_bytes),
            format_megabytes(MAX_UPLOAD_BYTES)
        )
    })?;

    stage(&format!(
        "audio is {} (over Together's {} limit); re-encoding to {} kHz mono Opus",
        format_megabytes(original_bytes),
        format_megabytes(MAX_UPLOAD_BYTES),
        UPLOAD_SHRINK_SAMPLE_RATE
    ));

    let output = TempAudio {
        path: unique_shrunk_audio_path(),
    };
    let mut command = Command::new(&ffmpeg);
    command
        .arg("-y")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(audio_path)
        .arg("-vn")
        .arg("-ac")
        .arg("1")
        .arg("-ar")
        .arg(UPLOAD_SHRINK_SAMPLE_RATE)
        .arg("-c:a")
        .arg("libopus")
        .arg("-b:a")
        .arg(UPLOAD_SHRINK_BITRATE)
        .arg(output.path())
        .stdin(Stdio::null());
    if config.print_command {
        stage(&format!("running {command:?}"));
    }

    let status = command
        .status()
        .with_context(|| format!("failed to run ffmpeg at `{}`", ffmpeg.display()))?;
    if !status.success() {
        bail!("ffmpeg failed to re-encode `{}`", audio_path.display());
    }

    let shrunk_bytes = fs::metadata(output.path())
        .with_context(|| format!("failed to inspect `{}`", output.path().display()))?
        .len();
    if shrunk_bytes > MAX_UPLOAD_BYTES {
        bail!(
            "re-encoded audio is still {} (limit {}); the recording likely exceeds \
             Together AI's 4-hour ceiling, so split it and transcribe each part",
            format_megabytes(shrunk_bytes),
            format_megabytes(MAX_UPLOAD_BYTES)
        );
    }

    stage(&format!(
        "re-encoded to {} ({} smaller)",
        format_megabytes(shrunk_bytes),
        format_shrink_ratio(original_bytes, shrunk_bytes)
    ));
    Ok(output)
}

fn unique_shrunk_audio_path() -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    env::temp_dir().join(format!("yt-transcript-upload-{pid}-{nanos}.opus"))
}

fn format_megabytes(bytes: u64) -> String {
    format!("{:.1} MB", bytes as f64 / 1_000_000.0)
}

fn format_shrink_ratio(original: u64, shrunk: u64) -> String {
    if shrunk == 0 {
        return "unknown".into();
    }
    format!("{:.1}x", original as f64 / shrunk as f64)
}

fn format_together_transcript(
    response: &TogetherTranscriptionResponse,
    timestamps: bool,
    diarise: bool,
) -> String {
    if !timestamps && !diarise {
        return response.text.clone();
    }

    if diarise && !response.speaker_segments.is_empty() {
        return response
            .speaker_segments
            .iter()
            .map(|segment| {
                format_together_segment(
                    &segment.text,
                    segment.start,
                    timestamps,
                    Some(&segment.speaker_id),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
    }

    response
        .segments
        .iter()
        .map(|segment| {
            let speaker = diarise.then(|| segment.speaker.as_deref().unwrap_or("UNKNOWN_SPEAKER"));
            format_together_segment(&segment.text, segment.start, timestamps, speaker)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_together_segment(
    text: &str,
    start: f64,
    timestamps: bool,
    speaker: Option<&str>,
) -> String {
    let text = text.trim();
    match (timestamps, speaker) {
        (true, Some(speaker)) => format!("[{}] {speaker} {text}", format_timestamp(start)),
        (true, None) => format!("[{}] {text}", format_timestamp(start)),
        (false, Some(speaker)) => format!("{speaker} {text}"),
        (false, None) => text.to_string(),
    }
}

fn format_timestamp(seconds: f64) -> String {
    let total_seconds = seconds.max(0.0).floor() as u64;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let seconds = total_seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

fn run_local_runtime(
    audio_path: &Path,
    model: &ModelProfile,
    runtime: ModelRuntime,
    config: &LocalTranscriptionConfig<'_>,
) -> Result<LocalTranscriptionResult> {
    if matches!(runtime, ModelRuntime::TogetherCloud) {
        bail!("Together cloud runtime cannot be launched through the local Python runner");
    }

    let script_path = ensure_transcriber_script()?;
    let result_path = unique_result_path();

    let mut command = Command::new(config.uv_path);
    command
        .arg("run")
        .arg("--python")
        .arg(config.python_version);
    for package in runtime.dependency_packages() {
        command.arg("--with").arg(package);
    }

    command
        .arg("--")
        .arg("python")
        .arg(&script_path)
        .arg("--runtime")
        .arg(runtime.as_script_value())
        .arg("--model-id")
        .arg(model.id)
        .arg("--audio-path")
        .arg(audio_path)
        .arg("--device")
        .arg(config.device.as_arg())
        .arg("--result-path")
        .arg(&result_path)
        .arg("--canary-max-new-tokens")
        .arg(config.canary_max_new_tokens.to_string())
        .arg("--mlx-chunk-seconds")
        .arg(config.mlx_chunk_seconds.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1");

    if config.transcribe_progress {
        command.arg("--transcribe-progress");
    }

    if let Some(mlx_model_id) = model.mlx_model_id {
        command.arg("--mlx-model-id").arg(mlx_model_id);
    }

    if let Some(token) = config.hf_token {
        command.env("HF_TOKEN", token);
        command.env("HUGGING_FACE_HUB_TOKEN", token);
    }

    if config.print_command {
        eprintln!("{}", render_command(&command));
    }

    let status = command
        .status()
        .with_context(|| format!("failed to execute `{}`", config.uv_path))?;
    if !status.success() {
        bail!("local transcription runtime failed with status {status}");
    }

    let raw = fs::read(&result_path).with_context(|| {
        format!(
            "failed to read local transcription output `{}`",
            result_path.display()
        )
    })?;
    let result: LocalTranscriptionResult = serde_json::from_slice(&raw).with_context(|| {
        format!(
            "invalid transcription output JSON at `{}`",
            result_path.display()
        )
    })?;
    let _ = fs::remove_file(&result_path);
    Ok(result)
}

fn ensure_transcriber_script() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("yt-transcript");
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create temp directory `{}`", dir.display()))?;

    let script_path = dir.join("local_transcribe.py");
    let should_write = match fs::read_to_string(&script_path) {
        Ok(existing) => existing != LOCAL_TRANSCRIBE_SCRIPT,
        Err(_) => true,
    };

    if should_write {
        fs::write(&script_path, LOCAL_TRANSCRIBE_SCRIPT).with_context(|| {
            format!(
                "failed to write transcriber script `{}`",
                script_path.display()
            )
        })?;
    }

    Ok(script_path)
}

fn unique_result_path() -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);

    std::env::temp_dir().join(format!("yt-transcript-result-{pid}-{nanos}.json"))
}

fn build_transcript_path(cli: &Cli, meta: &VideoMeta) -> PathBuf {
    if let Some(path) = &cli.transcript_path {
        return path.clone();
    }

    cli.output_dir
        .join(format!("{}-{}.txt", meta.safe_title, meta.id))
}

fn sanitize_filename(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut previous_underscore = false;

    for ch in raw.chars() {
        let normalized = if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            ch
        } else {
            '_'
        };

        if normalized == '_' {
            if previous_underscore {
                continue;
            }
            previous_underscore = true;
        } else {
            previous_underscore = false;
        }

        out.push(normalized);
    }

    out.trim_matches('_').to_string()
}

fn stage(message: &str) {
    eprintln!("[yt-transcript] {message}");
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs_f64() >= 1.0 {
        return format!("{:.2}s", duration.as_secs_f64());
    }

    format!("{:.0}ms", duration.as_secs_f64() * 1000.0)
}

fn try_wav_duration_seconds(path: &Path) -> Option<f64> {
    let mut file = fs::File::open(path).ok()?;

    let mut riff = [0_u8; 12];
    file.read_exact(&mut riff).ok()?;
    if &riff[0..4] != b"RIFF" || &riff[8..12] != b"WAVE" {
        return None;
    }

    let mut byte_rate: Option<u32> = None;
    let mut data_size: Option<u32> = None;

    loop {
        let mut header = [0_u8; 8];
        if file.read_exact(&mut header).is_err() {
            break;
        }

        let chunk_id = &header[0..4];
        let chunk_size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let chunk_size_u64 = u64::from(chunk_size);

        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                return None;
            }
            let mut fmt = vec![0_u8; chunk_size as usize];
            file.read_exact(&mut fmt).ok()?;
            byte_rate = Some(u32::from_le_bytes([fmt[8], fmt[9], fmt[10], fmt[11]]));
        } else if chunk_id == b"data" {
            data_size = Some(chunk_size);
            file.seek(SeekFrom::Current(i64::try_from(chunk_size_u64).ok()?))
                .ok()?;
        } else {
            file.seek(SeekFrom::Current(i64::try_from(chunk_size_u64).ok()?))
                .ok()?;
        }

        if chunk_size % 2 == 1 {
            file.seek(SeekFrom::Current(1)).ok()?;
        }

        if byte_rate.is_some() && data_size.is_some() {
            break;
        }
    }

    let byte_rate = byte_rate?;
    let data_size = data_size?;
    if byte_rate == 0 {
        return None;
    }
    Some(f64::from(data_size) / f64::from(byte_rate))
}

fn render_command(command: &Command) -> String {
    let mut full = Vec::with_capacity(1 + command.get_args().count());
    full.push(shell_escape(command.get_program()));
    full.extend(command.get_args().map(shell_escape));
    full.join(" ")
}

fn shell_escape(value: &std::ffi::OsStr) -> String {
    let text = value.to_string_lossy();

    if text
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "-._/:=%".contains(c))
    {
        return text.into_owned();
    }

    let escaped = text.replace('"', "\\\"");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn resolves_default_model_by_id() {
        let resolved = resolve_model(DEFAULT_MODEL_ID).expect("default model should resolve");
        assert_eq!(resolved.id, DEFAULT_MODEL_ID);
    }

    #[test]
    fn resolves_model_by_alias_case_insensitive() {
        let resolved = resolve_model("Canary").expect("canary alias should resolve");
        assert_eq!(resolved.id, "nvidia/canary-qwen-2.5b");
    }

    #[test]
    fn does_not_resolve_display_name_with_spaces() {
        let resolved = resolve_model("NVIDIA Canary Qwen 2.5B");
        assert!(resolved.is_none());
    }

    #[test]
    fn render_command_quotes_args_with_spaces() {
        let mut command = Command::new("yt-dlp");
        command
            .arg("--model")
            .arg(OsString::from("NVIDIA Canary Qwen 2.5B"));

        let rendered = render_command(&command);
        assert!(rendered.contains("\"NVIDIA Canary Qwen 2.5B\""));
    }

    #[test]
    fn sanitizes_filename_to_ascii_safe() {
        let out = sanitize_filename("Hello, world! (v2)");
        assert_eq!(out, "Hello_world_v2");
    }

    #[test]
    fn formats_together_segments_with_timestamps_and_diarisation() {
        let response = TogetherTranscriptionResponse {
            duration: 65.5,
            text: "Plain transcript".to_string(),
            segments: vec![
                TogetherSegment {
                    text: " First line ".to_string(),
                    start: 1.9,
                    speaker: Some("SPEAKER_0".to_string()),
                },
                TogetherSegment {
                    text: "Second line".to_string(),
                    start: 65.2,
                    speaker: Some("SPEAKER_1".to_string()),
                },
            ],
            speaker_segments: Vec::new(),
        };

        assert_eq!(
            format_together_transcript(&response, true, true),
            "[00:00:01] SPEAKER_0 First line\n[00:01:05] SPEAKER_1 Second line"
        );
        assert_eq!(
            format_together_transcript(&response, false, true),
            "SPEAKER_0 First line\nSPEAKER_1 Second line"
        );
        assert_eq!(
            format_together_transcript(&response, false, false),
            "Plain transcript"
        );
    }

    #[test]
    fn local_flag_selects_the_exclusive_transcription_backend() {
        assert_eq!(
            transcription_backend(false),
            TranscriptionBackend::TogetherCloud
        );
        assert_eq!(transcription_backend(true), TranscriptionBackend::Local);

        let model = resolve_model(DEFAULT_MODEL_ID).expect("default model should resolve");
        assert!(
            runtime_candidates(model)
                .iter()
                .all(|runtime| *runtime != ModelRuntime::TogetherCloud)
        );
    }

    #[test]
    fn models_list_works_when_diarize_flag_is_present() {
        let cli = Cli::try_parse_from(["yt-transcript", "--diarize", "models", "list"])
            .expect("CLI arguments should parse");
        assert!(cli.diarize);
        let command = cli.command.expect("models subcommand should be present");
        assert!(handle_command(command).is_ok());
    }

    #[test]
    fn upload_limit_matches_together_direct_upload_ceiling() {
        assert_eq!(MAX_UPLOAD_BYTES, 80_000_000);
    }

    #[test]
    fn shrink_settings_keep_four_hours_under_the_upload_limit() {
        // Together caps a single request at 4 hours of audio. Confirm the chosen
        // bitrate keeps even that worst case comfortably inside the byte limit,
        // so `shrink_for_upload` never needs a second pass.
        let bitrate_bits_per_second: u64 = 32_000;
        let four_hours_seconds: u64 = 4 * 60 * 60;
        let worst_case_bytes = bitrate_bits_per_second / 8 * four_hours_seconds;
        assert!(
            worst_case_bytes < MAX_UPLOAD_BYTES,
            "4h at {UPLOAD_SHRINK_BITRATE} is {worst_case_bytes} bytes, over the limit"
        );
    }

    #[test]
    fn shrink_sample_rate_matches_the_model_input_rate() {
        // Parakeet resamples to 16 kHz, so re-encoding above that only adds bytes.
        assert_eq!(UPLOAD_SHRINK_SAMPLE_RATE, "16000");
    }

    #[test]
    fn formats_sizes_and_ratios_for_stage_logs() {
        assert_eq!(format_megabytes(80_000_000), "80.0 MB");
        assert_eq!(format_megabytes(0), "0.0 MB");
        assert_eq!(format_shrink_ratio(100_000_000, 25_000_000), "4.0x");
        assert_eq!(format_shrink_ratio(100, 0), "unknown");
    }

    #[test]
    fn temp_audio_is_removed_on_drop() {
        let path = unique_shrunk_audio_path();
        fs::write(&path, b"placeholder").expect("temp file should be writable");
        assert!(path.is_file());
        drop(TempAudio { path: path.clone() });
        assert!(!path.exists(), "TempAudio should delete its file on drop");
    }

    fn unique_test_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let dir =
            env::temp_dir().join(format!("yt-transcript-test-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).expect("test dir should be creatable");
        dir
    }

    fn format_selector<S: AsRef<str>>(args: &[S]) -> String {
        args.windows(2)
            .find(|pair| pair[0].as_ref() == "-f")
            .map(|pair| pair[1].as_ref().to_string())
            .expect("args should include a format selector")
    }

    #[test]
    fn cloud_mode_falls_back_to_a_muxed_stream() {
        // Regression: cloud mode used a bare `bestaudio` selector, so any site
        // offering muxed streams only (Dailymotion, for one) failed with
        // "Requested format is not available".
        let model = resolve_model(DEFAULT_MODEL_ID).expect("default model should resolve");
        let args = audio_download_args(model, true, "ffmpeg").expect("cloud args should build");

        assert_eq!(
            format_selector(&args),
            concat!("bestaudio/", muxed_fallback!())
        );
        assert!(args.iter().any(|arg| arg == "--extract-audio"));
    }

    #[test]
    fn muxed_fallback_prefers_h264_before_plain_best() {
        // Regression: TikTok's H.265 muxed files are listed with AAC audio but
        // often have no audio track, so a bare `best` fallback downloaded a
        // silent file and ffprobe failed. Every selector must try H.264 first.
        let model = resolve_model(DEFAULT_MODEL_ID).expect("default model should resolve");
        let cloud = audio_download_args(model, true, "ffmpeg").expect("cloud args should build");
        let local = audio_download_args(model, false, "ffmpeg").expect("local args should build");
        let video = video_download_args();

        for selector in [
            format_selector(&cloud),
            format_selector(&local),
            format_selector(&video),
        ] {
            assert!(
                selector.ends_with("/best[vcodec^=h264]/best[vcodec^=avc1]/best"),
                "selector `{selector}` should prefer H.264 muxed streams"
            );
        }
    }

    #[test]
    fn cloud_mode_remuxes_instead_of_re_encoding() {
        // The fallback now runs for every cloud download, so it must keep the
        // source codec rather than re-encode lossily.
        let model = resolve_model(DEFAULT_MODEL_ID).expect("default model should resolve");
        let args = audio_download_args(model, true, "ffmpeg").expect("cloud args should build");
        let format = args
            .windows(2)
            .find(|pair| pair[0] == "--audio-format")
            .map(|pair| pair[1].clone())
            .expect("cloud args should set an audio format");

        assert_eq!(format, "best");
    }

    #[test]
    fn local_mode_uses_the_model_format_chain() {
        let model = resolve_model(DEFAULT_MODEL_ID).expect("default model should resolve");
        let args = audio_download_args(model, false, "ffmpeg").expect("local args should build");

        assert_eq!(format_selector(&args), model.yt_dlp_format);
        assert!(
            args.iter()
                .any(|arg| arg == "ffmpeg:-ac 1 -ar 16000 -sample_fmt s16")
        );
    }

    #[test]
    fn video_args_request_the_best_stream_merged_to_mp4() {
        let args = video_download_args();

        assert_eq!(
            format_selector(&video_download_args()),
            concat!("bestvideo*+bestaudio/", muxed_fallback!())
        );
        assert!(
            args.windows(2)
                .any(|pair| pair[0] == "--merge-output-format" && pair[1] == "mp4")
        );
    }

    #[test]
    fn keep_video_flag_parses() {
        let cli = Cli::try_parse_from(["yt-transcript", "--keep-video", "https://example.com/v"])
            .expect("CLI arguments should parse");
        assert!(cli.keep_video);
    }

    #[test]
    fn finds_the_kept_video_not_the_audio() {
        let dir = unique_test_dir();
        let base = "Title-xaynljy";
        fs::write(dir.join(format!("{base}.m4a")), b"audio").expect("audio should be writable");
        let expected = dir.join(format!("{base}-video.mp4"));
        fs::write(&expected, b"video").expect("video should be writable");

        let found = find_downloaded_video(&dir, base).expect("video should be found");
        assert_eq!(found, expected);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reports_a_missing_video_instead_of_settling_for_audio() {
        let dir = unique_test_dir();
        let base = "Title-xaynljy";
        fs::write(dir.join(format!("{base}.m4a")), b"audio").expect("audio should be writable");

        assert!(find_downloaded_video(&dir, base).is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn kept_video_is_ignored_by_delete_audio() {
        let dir = unique_test_dir();
        let audio = dir.join("Title-xaynljy.m4a");
        let video = dir.join("Title-xaynljy-video.mp4");
        fs::write(&audio, b"audio").expect("audio should be writable");
        fs::write(&video, b"video").expect("video should be writable");

        remove_audio_file(&audio).expect("audio should be removable");
        assert!(!audio.exists());
        assert!(
            video.is_file(),
            "--delete-audio must not touch the kept video"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
