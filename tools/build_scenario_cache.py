"""Build an offline cache of scenario text embeddings for / 3AE-5.

Reads ``train_data.csv``, deduplicates the ``scenario`` column, encodes
each unique scenario once with a frozen sentence-transformers model,
and writes the resulting ``{text: embedding}`` dict to a ``.pt`` file.

Usage (unmasked)::

    python -m tools.build_scenario_cache \
        --csv data/diema_challenge/raw/train_data.csv \
        --out output/scenario_text_cache.pt \
        --model sentence-transformers/all-MiniLM-L6-v2

Usage (emotion-masked)::

    python -m tools.build_scenario_cache \
        --csv data/diema_challenge/raw/train_data.csv \
        --out output/scenario_text_cache_masked.pt \
        --mask emotion \
        --forbidden configs/leakage/forbidden_tokens.yaml

The cache is a single ``torch.save`` payload with structure::

    {"embeddings": dict[str, Tensor], "meta": dict[str, str]}

Consumers load it via :func:`diema.features.text_features.load_text_cache`.
When ``--mask emotion`` is passed, the cache also stores a side-by-side
``"mask_report"`` payload summarizing how many scenarios were touched so
callers can validate against the masking ablation (a00 vs a10).

See: (emotion lexicon masking)
Related: diema/features/text_features.py, configs/leakage/forbidden_tokens.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

from diema.features.emotion_mask import (
    MASK_TOKEN,
    compile_emotion_mask_regex,
    load_forbidden_tokens,
    mask_emotion_aliases,
)
from diema.features.text_features import (
    DEFAULT_TEXT_ENCODER,
    ScenarioTextEncoder,
    save_text_cache,
)

# Back-compat private aliases — prefer diema.features.emotion_mask for new code.
_compile_emotion_mask_regex = compile_emotion_mask_regex
_load_forbidden_tokens = load_forbidden_tokens

__all__ = [
    "MASK_TOKEN",
    "mask_emotion_aliases",
    "build_cache",
]


def build_cache(
    csv_path: str | Path,
    out_path: str | Path,
    model_name: str = DEFAULT_TEXT_ENCODER,
    device: str = "cpu",
    mask: str | None = None,
    forbidden_yaml: str | Path | None = None,
) -> tuple[int, int]:
    """Build and write the scenario text cache.

    Args:
        csv_path: path to ``train_data.csv``.
        out_path: output ``.pt`` path.
        model_name: sentence-transformers checkpoint.
        device: encoder device (``cpu`` is fine; cache is offline).
        mask: None for no masking (default) or "emotion" for
             emotion-alias masking (a00 default).
        forbidden_yaml: path to the forbidden-tokens YAML. Required when
            ``mask="emotion"``.

    Returns:
        Tuple ``(num_rows, num_unique_scenarios_stored)``. With masking,
        ``num_unique_scenarios_stored`` counts unique *post-masking* texts,
        which may be smaller than the unmasked unique count when different
        scenarios collapse to the same masked form.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"train CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "scenario" not in df.columns:
        raise ValueError(
            f"{csv_path} does not contain a 'scenario' column; "
            "/3AE-5 text supervision expects the DIEM-A train CSV"
        )

    scenarios = df["scenario"].dropna().astype(str).tolist()

    mask_report: dict[str, str] = {}
    if mask == "emotion":
        if forbidden_yaml is None:
            forbidden_yaml = Path("configs/leakage/forbidden_tokens.yaml")
        forbidden = _load_forbidden_tokens(Path(forbidden_yaml))
        aliases = forbidden.get("emotion_aliases", [])
        if not aliases:
            raise ValueError(
                f"emotion_aliases missing / empty in {forbidden_yaml}; "
                " masking requires the G-12labels list"
            )
        regex = _compile_emotion_mask_regex(aliases)
        total_hits = 0
        num_touched = 0
        masked_scenarios: list[str] = []
        for text in scenarios:
            masked, hits = mask_emotion_aliases(text, regex)
            masked_scenarios.append(masked)
            total_hits += hits
            if hits > 0:
                num_touched += 1
        scenarios = masked_scenarios
        mask_report = {
            "mask_mode": "emotion",
            "emotion_aliases_count": str(len(aliases)),
            "rows_touched": str(num_touched),
            "total_hits": str(total_hits),
            "forbidden_yaml": str(forbidden_yaml),
            "mask_token": MASK_TOKEN,
        }
    elif mask is not None:
        raise ValueError(f"unknown mask mode {mask!r}; use None or 'emotion'")

    unique = sorted(set(scenarios))

    encoder = ScenarioTextEncoder(model_name=model_name, device=device)
    # Warm up + encode all unique strings in one batched call. The
    # encoder always caches the raw (unnormalized) vectors, regardless
    # of the ``normalize`` argument, so the on-disk cache is a safe
    # starting point for either normalized or unnormalized consumers.
    encoder.encode(unique, normalize=True)
    cache = encoder.cache_snapshot()

    meta = {
        "model_name": model_name,
        "device": device,
        "num_rows": str(len(scenarios)),
        "num_unique": str(len(unique)),
        "embedding_dim": str(encoder.embedding_dim()),
        "generated_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_csv": str(csv_path),
        **mask_report,
    }

    out_path = Path(out_path)
    save_text_cache(out_path, cache, meta=meta)
    return len(scenarios), len(unique)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenario text embedding cache.")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/diema_challenge/raw/train_data.csv",
        help="Path to train_data.csv (must contain a 'scenario' column).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/scenario_text_cache.pt",
        help="Output path for the .pt cache file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TEXT_ENCODER,
        help="Sentence-transformers checkpoint name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the encoder. CPU is fine; the cache is built once offline.",
    )
    parser.add_argument(
        "--mask",
        choices=[None, "emotion"],
        default=None,
        help="None (default) or 'emotion' (lexicon masking).",
    )
    parser.add_argument(
        "--forbidden",
        type=str,
        default="configs/leakage/forbidden_tokens.yaml",
        help="Path to forbidden-tokens YAML (only used when --mask=emotion).",
    )
    args = parser.parse_args()

    n_rows, n_unique = build_cache(
        csv_path=args.csv,
        out_path=args.out,
        model_name=args.model,
        device=args.device,
        mask=args.mask,
        forbidden_yaml=args.forbidden,
    )
    suffix = f" (mask={args.mask})" if args.mask else ""
    print(
        f"Wrote cache to {args.out}: {n_unique} unique scenarios "
        f"covering {n_rows} train rows{suffix}."
    )


if __name__ == "__main__":
    main()
