#!/usr/bin/env bash
# Install yt-transcript from this checkout on macOS or Linux, plus its
# runtime tools and the Claude Code skill. Re-running is safe.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skills_dir="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}"

need() { command -v "$1" >/dev/null 2>&1; }

if ! need cargo; then
  echo "cargo is required: install Rust from https://rustup.rs" >&2
  exit 1
fi

if ! need yt-dlp; then
  if need uv; then
    uv tool install yt-dlp
  elif need brew; then
    brew install yt-dlp
  else
    echo "install yt-dlp (uv tool install yt-dlp, or brew install yt-dlp)" >&2
    exit 1
  fi
fi

if ! need ffmpeg; then
  if need brew; then
    brew install ffmpeg
  elif need apt-get; then
    sudo apt-get install -y ffmpeg
  else
    echo "install ffmpeg with your package manager" >&2
    exit 1
  fi
fi

cargo install --locked --path "$repo_dir"

mkdir -p "$skills_dir"
ln -sfn "$repo_dir/skills/yt-transcript" "$skills_dir/yt-transcript"

echo "installed: $(command -v yt-transcript || echo "$HOME/.cargo/bin/yt-transcript")"
echo "skill: $skills_dir/yt-transcript"
echo "set TOGETHER_API_KEY before transcribing"
