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

const MODELS: [ModelProfile; 2] = [
    ModelProfile {
        id: "nvidia/parakeet-tdt-0.6b-v3",
        display_name: "NVIDIA Parakeet TDT 0.6B v3",
        aliases: &["parakeet", "parakeet-v3"],
        notes: "Default. Optimized for high-throughput multilingual transcription.",
        yt_dlp_format: "bestaudio[acodec*=opus]/bestaudio[abr>=128]/bestaudio/best",
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
        yt_dlp_format: "bestaudio[asr>=44100]/bestaudio[abr>=160]/bestaudio/best",
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
    command.arg("--no-playlist");

    if config.cloud_mode {
        command.arg("-f").arg("bestaudio");
    } else {
        let postprocessor_args = format!(
            "ffmpeg:-ac {} -ar {} -sample_fmt s16",
            model.channels, model.sample_rate_hz
        );
        let ffmpeg_location = resolve_executable_path(config.ffmpeg_path);
        let ffmpeg_arg_is_path = Path::new(config.ffmpeg_path).components().count() > 1;

        if ffmpeg_arg_is_path && ffmpeg_location.is_none() {
            bail!(
                "ffmpeg path `{}` does not exist or is not executable",
                config.ffmpeg_path
            );
        }

        command
            .arg("--extract-audio")
            .arg("--audio-format")
            .arg(model.output_format)
            .arg("--audio-quality")
            .arg("0")
            .arg("--postprocessor-args")
            .arg(postprocessor_args)
            .arg("-f")
            .arg(model.yt_dlp_format);

        if let Some(path) = ffmpeg_location {
            command.arg("--ffmpeg-location").arg(path);
        }
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
    const MAX_UPLOAD_BYTES: u64 = 500_000_000;
    const ENDPOINT: &str = "https://api.together.ai/v1/audio/transcriptions";

    let api_key = config
        .together_api_key
        .context("Together cloud runtime requires TOGETHER_API_KEY")?;
    let together_model_id = model
        .together_model_id
        .context("the selected model is not available on Together AI")?;
    let metadata = fs::metadata(audio_path)
        .with_context(|| format!("failed to inspect audio file `{}`", audio_path.display()))?;
    if metadata.len() > MAX_UPLOAD_BYTES {
        bail!(
            "audio file `{}` is too large for Together AI's 500 MB direct upload limit",
            audio_path.display()
        );
    }

    let file = fs::File::open(audio_path)
        .with_context(|| format!("failed to open audio file `{}`", audio_path.display()))?;
    let file_name = audio_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("audio")
        .to_string();
    let file_part = reqwest::blocking::multipart::Part::reader(file).file_name(file_name);
    let diarise = config.diarize || config.speakers.is_some();
    let mut form = reqwest::blocking::multipart::Form::new()
        .part("file", file_part)
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
}
