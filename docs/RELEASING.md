# Releasing yt-transcript

This runbook is the source of truth for shipping a new `yt-transcript` version and making it available via Homebrew.

## Repos

- App repo: `malcomsonbrothers/yt-transcript`
- Tap repo: `malcomsonbrothers/homebrew-tap`

Local paths used in examples:

- `/Users/will/Documents/Areas/CommandLinePrograms/yt-transcript`
- `/Users/will/Documents/Areas/CommandLinePrograms/homebrew-tap`

## 1) Bump version in app repo

In `yt-transcript`:

```bash
cd /Users/will/Documents/Areas/CommandLinePrograms/yt-transcript
```

Update both files to the same version:

- `Cargo.toml` package version
- `Cargo.lock` `yt-transcript` package version

Validate:

```bash
cargo check
cargo test
rg -n '^version = "0\\.[0-9]+\\.[0-9]+"$' Cargo.toml Cargo.lock
```

Commit and push:

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: bump crate version to 0.x.y"
git push origin master
```

## 2) Tag release and publish binaries

Create and push a tag in the app repo:

```bash
git tag v0.x.y
git push origin v0.x.y
```

The workflow `.github/workflows/release-binaries.yml` builds and publishes:

- `yt-transcript-v0.x.y-macos-arm64.tar.gz`
- `yt-transcript-v0.x.y-linux-x86_64.tar.gz`
- matching `.sha256` files

Watch status:

```bash
gh run list --repo malcomsonbrothers/yt-transcript --workflow "Release Binaries" --limit 5
gh run watch <run-id> --repo malcomsonbrothers/yt-transcript --exit-status
```

## 3) Get release SHA256 values

After the workflow succeeds:

```bash
gh release view v0.x.y --repo malcomsonbrothers/yt-transcript --json assets,url
```

Use the `digest` for:

- `yt-transcript-v0.x.y-macos-arm64.tar.gz`
- `yt-transcript-v0.x.y-linux-x86_64.tar.gz`

## 4) Update Homebrew tap formula

In `homebrew-tap`:

```bash
cd /Users/will/Documents/Areas/CommandLinePrograms/homebrew-tap
```

Edit `Formula/yt-transcript.rb`:

- set `version "0.x.y"`
- update macOS arm64 `sha256`
- update Linux x86_64 `sha256`

Commit and push:

```bash
git add Formula/yt-transcript.rb
git commit -m "chore: bump yt-transcript to v0.x.y"
git push origin master
```

## 5) Verify install path (important)

Run:

```bash
brew update
brew upgrade malcomsonbrothers/tap/yt-transcript
yt-transcript --version
```

Expected: CLI version matches the formula version (`0.x.y`).

If version mismatches, check PATH conflicts:

```bash
which yt-transcript
brew list --versions yt-transcript
```

## Common pitfalls

- Bumping `Cargo.toml` but not committing `Cargo.lock`.
  - This breaks release CI when building with `--locked`.
- Updating app release but forgetting to bump `homebrew-tap` formula.
- Tagging before confirming local tests pass.

## Quick checklist

- [ ] `Cargo.toml` and `Cargo.lock` are on the same version
- [ ] tests pass locally
- [ ] tag pushed (`v0.x.y`)
- [ ] release workflow succeeded
- [ ] tap formula version + checksums updated
- [ ] `brew upgrade ...` works
- [ ] `yt-transcript --version` prints `0.x.y`
