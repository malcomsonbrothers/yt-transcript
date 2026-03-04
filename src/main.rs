use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde::Deserialize;

const DEFAULT_MODEL_ID: &str = "nvidia/parakeet-tdt-0.6b-v3";
const LOCAL_TRANSCRIBE_SCRIPT: &str = include_str!("nemo_transcribe.py");

#[derive(Parser, Debug)]
#[command(
    name = "yt-transcript",
    version,
    about = "Download YouTube audio and transcribe it locally with the selected model"
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

    /// Device selection for local inference.
    #[arg(long, value_enum, default_value_t = DeviceMode::Auto)]
    device: DeviceMode,

    /// max_new_tokens used for Canary generation.
    #[arg(long, default_value_t = 256)]
    canary_max_new_tokens: u16,

    /// Disable yt-dlp download progress output.
    #[arg(long)]
    no_download_progress: bool,

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

#[derive(Debug, Clone, Copy)]
enum ModelRuntime {
    ParakeetMlx,
    ParakeetNemo,
    CanaryNemo,
}

impl ModelRuntime {
    fn as_script_value(self) -> &'static str {
        match self {
            Self::ParakeetMlx => "parakeet_mlx",
            Self::ParakeetNemo => "parakeet_nemo",
            Self::CanaryNemo => "canary_nemo",
        }
    }

    fn short_name(self) -> &'static str {
        match self {
            Self::ParakeetMlx => "parakeet-mlx",
            Self::ParakeetNemo => "parakeet-nemo",
            Self::CanaryNemo => "canary-nemo",
        }
    }

    fn dependency_packages(self) -> &'static [&'static str] {
        match self {
            Self::ParakeetMlx => &["parakeet-mlx"],
            Self::ParakeetNemo | Self::CanaryNemo => &["torch", "nemo_toolkit[asr]"],
        }
    }

    fn description(self) -> &'static str {
        match self {
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
    print_command: bool,
    no_download_progress: bool,
}

#[derive(Debug)]
struct LocalTranscriptionConfig<'a> {
    uv_path: &'a str,
    python_version: &'a str,
    hf_token: Option<&'a str>,
    device: DeviceMode,
    canary_max_new_tokens: u16,
    print_command: bool,
}

#[derive(Debug, Deserialize)]
struct LocalTranscriptionResult {
    transcript: String,
    device: String,
    model_id: String,
    runtime: String,
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
    },
];

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(command) = cli.command {
        return handle_command(command);
    }

    let url = cli
        .url
        .as_deref()
        .context("a URL is required unless using a subcommand")?;

    let model = resolve_model(&cli.model).ok_or_else(|| {
        anyhow!(
            "unknown model `{}`; run `yt-transcript models list`",
            cli.model
        )
    })?;

    fs::create_dir_all(&cli.output_dir).with_context(|| {
        format!(
            "failed to create output directory `{}`",
            cli.output_dir.display()
        )
    })?;

    stage(&format!("resolving video metadata for {url}"));
    let video_meta = fetch_video_metadata(url, &cli.yt_dlp_path, cli.print_command)?;

    stage(&format!("downloading audio for model {}", model.id));
    let download_config = DownloadConfig {
        output_dir: &cli.output_dir,
        yt_dlp_path: &cli.yt_dlp_path,
        ffmpeg_path: &cli.ffmpeg_path,
        print_command: cli.print_command,
        no_download_progress: cli.no_download_progress,
    };
    let audio_path = download_audio(url, model, &video_meta, &download_config)?;

    stage(&format!(
        "transcribing locally with {} ({})",
        model.display_name, model.id
    ));
    let local_transcription = transcribe_audio_local(
        &audio_path,
        model,
        &LocalTranscriptionConfig {
            uv_path: &cli.uv_path,
            python_version: &cli.python_version,
            hf_token: cli.hf_token.as_deref(),
            device: cli.device,
            canary_max_new_tokens: cli.canary_max_new_tokens,
            print_command: cli.print_command,
        },
    )?;

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

    fs::write(&transcript_path, local_transcription.transcript).with_context(|| {
        format!(
            "failed to write transcript to `{}`",
            transcript_path.display()
        )
    })?;

    if cli.delete_audio {
        fs::remove_file(&audio_path)
            .with_context(|| format!("failed to delete audio file `{}`", audio_path.display()))?;
    }

    stage("done");
    println!("audio_file={}", audio_path.display());
    println!("transcript_file={}", transcript_path.display());
    println!("device={}", local_transcription.device);
    println!("runtime={}", local_transcription.runtime);

    Ok(())
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
    if cfg!(target_os = "macos") && model.mlx_model_id.is_some() {
        format!(
            "auto: {} -> {} fallback",
            ModelRuntime::ParakeetMlx.description(),
            ModelRuntime::ParakeetNemo.short_name(),
        )
    } else {
        model.runtime.description().to_string()
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

    let mut command = Command::new(config.yt_dlp_path);
    command
        .arg("--no-playlist")
        .arg("--extract-audio")
        .arg("--audio-format")
        .arg(model.output_format)
        .arg("--audio-quality")
        .arg("0")
        .arg("--postprocessor-args")
        .arg(postprocessor_args)
        .arg("-f")
        .arg(model.yt_dlp_format)
        .arg("-o")
        .arg(output_template)
        .arg(url)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if let Some(path) = ffmpeg_location {
        command.arg("--ffmpeg-location").arg(path);
    }

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
        bail!("yt-dlp download failed with status {status}");
    }

    if !output_audio.exists() {
        bail!(
            "expected audio output `{}` was not produced",
            output_audio.display()
        );
    }

    Ok(output_audio)
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
                    "runtime {} failed, trying fallback if available",
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

fn runtime_candidates(model: &ModelProfile) -> Vec<ModelRuntime> {
    match model.runtime {
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

fn run_local_runtime(
    audio_path: &Path,
    model: &ModelProfile,
    runtime: ModelRuntime,
    config: &LocalTranscriptionConfig<'_>,
) -> Result<LocalTranscriptionResult> {
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
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .env("PYTORCH_ENABLE_MPS_FALLBACK", "1");

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
}
