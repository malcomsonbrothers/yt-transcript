# yt-transcript

Rust CLI that downloads YouTube audio with `yt-dlp` and transcribes it with Together AI by default or local runtimes when requested.

## Runtime model path

- Default: uploads the downloaded audio to Together AI for hosted transcription. This path requires `TOGETHER_API_KEY` and never falls back to a local runtime.
- `--local`: keeps audio and inference local and never contacts Together AI.
  - Parakeet on macOS: tries MLX first (`uv run --with parakeet-mlx`), then falls back to NeMo.
  - Canary: uses NeMo (`uv run --with torch --with nemo_toolkit[asr]`). Canary is local-only and requires `--local`.
- Missing local model weights are downloaded automatically on first use and reused after that.

## Requirements

Every run that transcribes a video requires:

- `yt-dlp` on your `PATH` (or pass `--yt-dlp-path`)

(`yt-transcript models list` needs none of the tools below.)

Cloud transcription (the default) requires:

- `TOGETHER_API_KEY` (or pass `--together-api-key`)

Local transcription with `--local` also requires:

- `ffmpeg` on your `PATH` (or pass `--ffmpeg-path`)
- `uv` on your `PATH` (or pass `--uv-path`)

Optional for local transcription:

- `HF_TOKEN` (only needed for gated/private model downloads)

## Install

```bash
cargo install --path .
```

## Releasing

For the full release + Homebrew tap flow, see [docs/RELEASING.md](docs/RELEASING.md).

## Usage

Default cloud transcription with Parakeet v3:

```bash
export TOGETHER_API_KEY="your-api-key"
yt-transcript "https://www.youtube.com/watch?v=VIDEO_ID"
```

Pass the cloud key directly instead:

```bash
yt-transcript "https://www.youtube.com/watch?v=VIDEO_ID" --together-api-key "your-api-key"
```

Use local Parakeet inference:

```bash
yt-transcript "https://www.youtube.com/watch?v=VIDEO_ID" --local
```

Use the local-only Canary model:

```bash
yt-transcript "https://www.youtube.com/watch?v=VIDEO_ID" --local --model nvidia/canary-qwen-2.5b
```

Request cloud speaker diarisation, a fixed speaker count, and timestamps:

```bash
yt-transcript "https://www.youtube.com/watch?v=VIDEO_ID" --diarize --speakers 2 --timestamps
```

List supported model IDs and their available runtimes:

```bash
yt-transcript models list
```

## Cloud uploads and limits

The default path uploads downloaded audio to Together AI, a third-party service. Use `--local` if the audio must not leave your machine.

Together AI accepts at most 500 MB per upload and 4 hours of audio per transcription request.

## Device behaviour

These options apply only with `--local`:

- `--device auto` (default):
  - macOS Apple Silicon: prefers `mps`, then CPU fallback
  - other systems: prefers `cuda` when available, then CPU fallback
- You can force a device with `--device mps|cuda|cpu`.

## Output

- Cloud mode downloads the best audio stream in its source container and uploads it as-is: `<safe-title>-<video-id>.<source-extension>`
- Local mode converts audio to mono 16 kHz WAV: `<safe-title>-<video-id>.wav`
- Writes transcript text: `<safe-title>-<video-id>.txt`
- Prints:
  - `audio_file=...`
  - `transcript_file=...`
  - `device=...`
  - `runtime=...`

## Caching

For local transcription:

- Model weights are cached by Hugging Face under `~/.cache/huggingface/hub/...`
- `uv` package/runtime cache is under `~/.cache/uv/...`

## Progress / logging

- yt-dlp progress is shown by default
- Stage logs are printed to stderr (`[yt-transcript] ...`)
- Use `--no-download-progress` to hide yt-dlp progress
- Local MLX runtime can print chunked progress updates (`[local-transcribe] progress=...%`)
- Use `--no-transcribe-progress` to disable local transcription progress logs

## Timing output

At the end of each run, the CLI prints:

- `timing_metadata=...`
- `timing_download=...`
- `timing_transcription=...`
- `timing_write=...`
- `timing_total=...`
- `audio_duration=...` (when reported by Together AI or available from the local WAV)
- `transcription_speed=...x_realtime`

## Useful flags

Cloud/backend selection:

- `--together-api-key KEY` (alternatively set `TOGETHER_API_KEY`)
- `--local`
- `--diarize` (cloud-only speaker diarisation)
- `--speakers N` (cloud-only fixed speaker count; also enables diarisation)
- `--timestamps` (include segment timestamps in cloud transcripts)

Output and progress:

- `--output-dir ./out`
- `--transcript-path ./out/transcript.txt`
- `--delete-audio`
- `--print-command`
- `--no-download-progress`

Local runtime tuning:

- `--python-version 3.12`
- `--canary-max-new-tokens 256`
- `--mlx-chunk-seconds 120`
- `--no-transcribe-progress`
