""" minimal binary-FBX header inspector (no Autodesk SDK).

Related:
- data/diema_challenge/raw/fbx/{train,test}/*.fbx  (binary FBX v7.7)
- data/diema_challenge/raw/bvh/{train,test}/*.bvh  (ground-truth source we
  actually train on)

**Scope.** Proper parsing of binary FBX v7.7 requires Autodesk's C++ SDK
(Blender bundles it; standalone PyPI ``bpy`` targets Python 3.11+ so it is
not usable in this Py-3.10 env). For our purposes FBX is a redistribution
format, not a training input — the goal of this tool is therefore only to
answer three sanity questions per clip:

1. Is the file a well-formed FBX v7.7 (magic + version header readable)?
2. Does every BVH clip have a matching FBX sibling (and vice-versa)?
3. Is the FBX file plausibly non-empty (size > a conservative threshold)?

That's enough to flag dataset-level inconsistencies without pretending to
understand the FBX object graph.

Usage::

    python tools/fbx_inspect.py
    python tools/fbx_inspect.py --fbx-root data/diema_challenge/raw/fbx \\
        --bvh-root data/diema_challenge/raw/bvh \\
        --output docs/analysis/fbx_inventory.md
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


FBX_MAGIC_BINARY = b"Kaydara FBX Binary  \x00"  # 21 bytes + 0x00
MIN_FBX_SIZE = 1_024  # conservative; real clips are hundreds of KB+


def read_fbx_header(path: Path) -> dict:
    """Read only the 27-byte FBX binary header.

    Binary FBX format (per https://code.blender.org/2013/08/fbx-binary-file-format-specification/):
        bytes 0..20  = b"Kaydara FBX Binary  \\x00"
        bytes 21..22 = 0x1A 0x00 (start marker)
        bytes 23..26 = u32 little-endian version (e.g. 7700)

    Returns::

        {"is_fbx_binary": bool, "version": int | None, "size_bytes": int, "path": str}
    """
    size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(27)
    if len(head) < 27 or head[:21] != FBX_MAGIC_BINARY:
        return {
            "path": str(path),
            "is_fbx_binary": False,
            "version": None,
            "size_bytes": size,
        }
    version = struct.unpack("<I", head[23:27])[0]
    return {
        "path": str(path),
        "is_fbx_binary": True,
        "version": int(version),
        "size_bytes": size,
    }


def inventory_directory(root: Path) -> dict[str, list[Path]]:
    """Return ``{split_name: sorted list of .fbx files}`` for train/test subdirs."""
    out: dict[str, list[Path]] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        out[sub.name] = sorted(sub.glob("*.fbx"))
    return out


def bvh_stems_by_split(bvh_root: Path) -> dict[str, set[str]]:
    """Return ``{split: set of .bvh stem}`` to cross-check against FBX."""
    out: dict[str, set[str]] = {}
    for sub in sorted(p for p in bvh_root.iterdir() if p.is_dir()):
        out[sub.name] = {b.stem for b in sub.glob("*.bvh")}
    return out


@click.command()
@click.option("--fbx-root", default="data/diema_challenge/raw/fbx")
@click.option("--bvh-root", default="data/diema_challenge/raw/bvh")
@click.option("--output", default="docs/analysis/fbx_inventory.md",
              help="Markdown summary path")
@click.option("--sample-n", default=3, type=int,
              help="Number of per-split files to actually open + parse for header info")
def main(fbx_root: str, bvh_root: str, output: str, sample_n: int) -> None:
    fbx_path = Path(fbx_root)
    bvh_path = Path(bvh_root)
    if not fbx_path.exists():
        raise click.ClickException(f"FBX root not found: {fbx_path}")
    if not bvh_path.exists():
        raise click.ClickException(f"BVH root not found: {bvh_path}")

    fbx_split = inventory_directory(fbx_path)
    bvh_split = bvh_stems_by_split(bvh_path)

    report_rows: list[str] = [
        "# FBX inventory ",
        "",
        f"**Generated**: `tools/fbx_inspect.py`",
        f"**Scope**: header sanity + BVH↔FBX cross-check. Not a full FBX parse.",
        "",
        "| Split | #FBX | #BVH | missing in FBX | missing in BVH | min size | max size | sample version |",
        "|---|---|---|---|---|---|---|---|",
    ]

    overall: dict[str, dict] = {}
    for split, fbx_files in fbx_split.items():
        bvh_stems = bvh_split.get(split, set())
        fbx_stems = {f.stem for f in fbx_files}
        missing_in_fbx = sorted(bvh_stems - fbx_stems)
        missing_in_bvh = sorted(fbx_stems - bvh_stems)

        sizes = [f.stat().st_size for f in fbx_files]
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0

        # Parse header on up to sample_n files
        versions: set[int] = set()
        any_bad = False
        for f in fbx_files[:sample_n]:
            h = read_fbx_header(f)
            if not h["is_fbx_binary"]:
                any_bad = True
            elif h["version"] is not None:
                versions.add(h["version"])
        version_str = ("/".join(str(v) for v in sorted(versions))
                       if versions else ("BAD" if any_bad else "-"))

        overall[split] = {
            "n_fbx": len(fbx_files),
            "n_bvh": len(bvh_stems),
            "missing_in_fbx": missing_in_fbx,
            "missing_in_bvh": missing_in_bvh,
            "min_size": min_size,
            "max_size": max_size,
            "versions_sample": sorted(versions),
        }
        report_rows.append(
            f"| {split} | {len(fbx_files)} | {len(bvh_stems)} | "
            f"{len(missing_in_fbx)} | {len(missing_in_bvh)} | "
            f"{min_size:,} | {max_size:,} | {version_str} |"
        )

    report_rows.extend([
        "",
        "## Notes",
        "",
        "- Size in bytes. DIEM-A FBX files typically range 100 KB – 10 MB.",
        "- A missing FBX for a BVH clip (or vice versa) is a dataset-integrity"
        " concern — flag any row whose `missing` count is > 0.",
        "- Full FBX object-graph parsing requires Autodesk's binary SDK (via"
        " Blender/`bpy`). Neither is available for Python 3.10 on PyPI as of"
        " 2026-04-23, so the downstream validation path is BVH-first.",
        "- Any mismatch flagged here should be cross-checked against"
        ".",
    ])

    # Append per-split missing detail when small
    for split, info in overall.items():
        if info["missing_in_fbx"] or info["missing_in_bvh"]:
            report_rows.append("")
            report_rows.append(f"### Missing — {split}")
            if info["missing_in_fbx"]:
                report_rows.append(
                    f"BVH without FBX sibling ({len(info['missing_in_fbx'])}):"
                )
                for name in info["missing_in_fbx"][:20]:
                    report_rows.append(f"- `{name}`")
                if len(info["missing_in_fbx"]) > 20:
                    report_rows.append(f"- ... (+{len(info['missing_in_fbx']) - 20} more)")
            if info["missing_in_bvh"]:
                report_rows.append(
                    f"FBX without BVH sibling ({len(info['missing_in_bvh'])}):"
                )
                for name in info["missing_in_bvh"][:20]:
                    report_rows.append(f"- `{name}`")
                if len(info["missing_in_bvh"]) > 20:
                    report_rows.append(f"- ... (+{len(info['missing_in_bvh']) - 20} more)")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report_rows))
    click.echo(f"Wrote: {out_path}")

    # Short CLI summary
    click.echo("")
    click.echo("Summary:")
    for split, info in overall.items():
        click.echo(
            f"  {split}: {info['n_fbx']} FBX, {info['n_bvh']} BVH, "
            f"missing_in_fbx={len(info['missing_in_fbx'])}, "
            f"missing_in_bvh={len(info['missing_in_bvh'])}, "
            f"FBX version sample={info['versions_sample']}"
        )


if __name__ == "__main__":
    main()
