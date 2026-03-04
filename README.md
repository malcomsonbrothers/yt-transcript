# yt-transcript

Rust CLI that downloads model-compatible YouTube audio with `yt-dlp` and transcribes it locally.

## Runtime model path

- Local-only inference (no external transcription API calls).
- Runtime is selected automatically:
  - Parakeet on macOS: tries MLX first (`uv run --with parakeet-mlx`), then falls back to NeMo.
  - Canary: uses NeMo (`uv run --with torch --with nemo_toolkit[asr]`).
- Missing model weights are downloaded automatically on first use and reused after that.

## Requirements

- `yt-dlp` on your `PATH` (or pass `--yt-dlp-path`)
- `ffmpeg` on your `PATH` (or pass `--ffmpeg-path`)
- `uv` on your `PATH` (or pass `--uv-path`)

Optional:

- `HF_TOKEN` (only needed for gated/private model downloads)

## Install

```bash
cargo install --path .
```

## Usage

Default model (Parakeet v3):

```bash
yt-transcript "https://www.youtube.com/watch?v=VIDEO_ID"
```

Use Canary via model ID:

```bash
yt-transcript "https://www.youtube.com/watch?v=VIDEO_ID" --model nvidia/canary-qwen-2.5b
```

List supported model IDs:

```bash
yt-transcript models list
```

## Device behavior

- `--device auto` (default):
  - macOS Apple Silicon: prefers `mps`, then CPU fallback
  - other systems: prefers `cuda` when available, then CPU fallback
- You can force a device with `--device mps|cuda|cpu`.

## Output

- Downloads audio as mono 16kHz WAV: `<safe-title>-<video-id>.wav`
- Writes transcript text: `<safe-title>-<video-id>.txt`
- Prints:
  - `audio_file=...`
  - `transcript_file=...`
  - `device=...`
  - `runtime=...`

## Caching

- Model weights are cached by Hugging Face under `~/.cache/huggingface/hub/...`
- `uv` package/runtime cache is under `~/.cache/uv/...`

## Progress / logging

- yt-dlp progress is shown by default
- Stage logs are printed to stderr (`[yt-transcript] ...`)
- Use `--no-download-progress` to hide yt-dlp progress
- MLX runtime can print chunked progress updates (`[local-transcribe] progress=...%`)
- Use `--no-transcribe-progress` to disable transcription progress logs

## Timing output

At the end of each run, the CLI prints:

- `timing_metadata=...`
- `timing_download=...`
- `timing_transcription=...`
- `timing_write=...`
- `timing_total=...`
- `audio_duration=...` (when WAV duration is available)
- `transcription_speed=...x_realtime`

## Useful flags

- `--output-dir ./out`
- `--transcript-path ./out/transcript.txt`
- `--delete-audio`
- `--print-command`
- `--python-version 3.12`
- `--canary-max-new-tokens 256`
- `--mlx-chunk-seconds 120`
- `--no-transcribe-progress`
