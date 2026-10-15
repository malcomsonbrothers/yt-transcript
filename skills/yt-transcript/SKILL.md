---
name: yt-transcript
description: Transcribe a YouTube (or other yt-dlp supported) video to text with the yt-transcript CLI, using Together AI speech-to-text by default. Load when asked for the transcript, captions or spoken content of a video URL.
---

# yt-transcript

`yt-transcript` downloads a video's audio with `yt-dlp` and transcribes it
with Together AI (Parakeet v3 by default). Never use `--local` unless the
user asks for local inference.

## Run

```bash
yt-transcript "<url>" --output-dir <dir> --delete-audio --no-download-progress
```

- Needs `TOGETHER_API_KEY` in the environment (or `--together-api-key`).
  Where the key lives in a secret helper, wrap the command so the value is
  never printed, for example
  `agent-tool-secret exec --project <project> TOGETHER_API_KEY -- yt-transcript ...`.
- Add `--timestamps` for segment timestamps and `--diarize` (or
  `--speakers N`) for speaker labels.
- The transcript lands at `<dir>/<safe-title>-<video-id>.txt`; the CLI
  prints `transcript_file=...` on stdout. Use `--transcript-path` to pick
  the file name.

## Failures

- `Sign in to confirm you're not a bot` / `LOGIN_REQUIRED` from yt-dlp:
  YouTube is blocking this machine's IP. Retrying with other player clients
  does not help; run it from another network or ask the user.
- `requires TOGETHER_API_KEY`: the key is missing; report where it should
  go rather than falling back to `--local`.
