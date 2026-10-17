//! `--audio-file`: transcribe audio that is already on disk, skipping yt-dlp.
//!
//! YouTube stream links are signed for the IP that requested them, so audio
//! fetched on another machine cannot be re-downloaded on a bot-blocked host.
//! Copying the file over and passing it here is the supported route.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::sanitize_filename;

/// Check the file exists and suits the selected backend.
///
/// Together accepts the source container as-is. The local runtimes are fed
/// 16 kHz mono WAV by the download path, so `--local` takes WAV only.
pub(crate) fn validate(path: &Path, local: bool) -> Result<()> {
    if !path.is_file() {
        bail!("audio file `{}` does not exist", path.display());
    }
    let is_wav = path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"));
    if local && !is_wav {
        bail!(
            "`--local` with `--audio-file` needs 16 kHz mono WAV; convert `{}` first or drop `--local`",
            path.display()
        );
    }
    Ok(())
}

/// Default transcript path: the file's sanitised stem in the output directory.
pub(crate) fn transcript_path(output_dir: &Path, audio: &Path) -> PathBuf {
    let stem = audio
        .file_stem()
        .and_then(|stem| stem.to_str())
        .map(sanitize_filename)
        .filter(|stem| !stem.is_empty())
        .unwrap_or_else(|| "audio".to_string());
    output_dir.join(format!("{stem}.txt"))
}
