#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def resolve_device(requested: str, runtime: str) -> str:
    if requested != "auto":
        return requested

    if runtime == "parakeet_mlx":
        if platform.system() == "Darwin":
            return "mps"
        return "cpu"

    import torch

    if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def extract_text(value) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if hasattr(value, "text"):
        maybe_text = getattr(value, "text")
        if isinstance(maybe_text, str):
            return maybe_text.strip()

    return str(value).strip()


def transcribe_parakeet(model_id: str, audio_path: str, device: str) -> str:
    import torch
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name=model_id,
        map_location=torch.device(device),
    )
    output = model.transcribe([audio_path])
    if output and len(output) > 0:
        return extract_text(output[0])

    return ""


def transcribe_parakeet_mlx(
    model_id: str,
    mlx_model_id: str,
    audio_path: str,
    progress: bool,
    chunk_seconds: float,
) -> str:
    from parakeet_mlx import from_pretrained

    model = from_pretrained(mlx_model_id)

    if progress and chunk_seconds > 0:
        next_percent = 0

        def callback(current_position, total_position):
            nonlocal next_percent
            if total_position <= 0:
                return
            percent = int((float(current_position) / float(total_position)) * 100.0)
            if percent >= next_percent:
                print(f"[local-transcribe] progress={percent}%", file=sys.stderr)
                next_percent = min(100, next_percent + 5)

        result = model.transcribe(
            audio_path,
            chunk_duration=chunk_seconds,
            chunk_callback=callback,
        )
    else:
        result = model.transcribe(audio_path)

    return extract_text(getattr(result, "text", ""))


def transcribe_canary(model_id: str, audio_path: str, device: str, max_new_tokens: int) -> str:
    import torch
    from nemo.collections.speechlm2.models import SALM

    model = SALM.from_pretrained(model_id)
    model = model.to(device)

    answer_ids = model.generate(
        prompts=[
            [
                {
                    "role": "user",
                    "content": f"Transcribe the following: {model.audio_locator_tag}",
                    "audio": [audio_path],
                }
            ]
        ],
        max_new_tokens=max_new_tokens,
    )

    if answer_ids is None:
        return ""

    if hasattr(answer_ids, "__len__") and len(answer_ids) == 0:
        return ""

    first = answer_ids[0]
    if hasattr(first, "detach"):
        first = first.detach()
    if hasattr(first, "cpu"):
        first = first.cpu()

    text = model.tokenizer.ids_to_text(first)
    return extract_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local NeMo transcription runner")
    parser.add_argument(
        "--runtime",
        required=True,
        choices=["parakeet_mlx", "parakeet_nemo", "canary_nemo"],
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--mlx-model-id", default="mlx-community/parakeet-tdt-0.6b-v3")
    parser.add_argument("--audio-path", required=True)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
    )
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--canary-max-new-tokens", type=int, default=256)
    parser.add_argument("--mlx-chunk-seconds", type=float, default=120.0)
    parser.add_argument("--transcribe-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"audio file does not exist: {audio_path}", file=sys.stderr)
        return 2

    result_path = Path(args.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device, args.runtime)
    print(f"[local-transcribe] runtime={args.runtime} model={args.model_id} device={device}", file=sys.stderr)

    try:
        if args.runtime == "parakeet_mlx":
            transcript = transcribe_parakeet_mlx(
                args.model_id,
                args.mlx_model_id,
                str(audio_path),
                args.transcribe_progress,
                args.mlx_chunk_seconds,
            )
        elif args.runtime == "parakeet_nemo":
            transcript = transcribe_parakeet(args.model_id, str(audio_path), device)
        else:
            transcript = transcribe_canary(
                args.model_id,
                str(audio_path),
                device,
                args.canary_max_new_tokens,
            )
    except Exception as exc:  # pragma: no cover
        print(f"[local-transcribe] transcription failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "transcript": transcript,
        "device": device,
        "model_id": args.model_id,
        "runtime": args.runtime,
    }

    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
