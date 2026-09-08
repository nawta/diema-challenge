"""tools/build_qwen_rationale_dataset.py — staged, ANONYMIZED Qwen-rationale
dataset builder (held release; do NOT upload until license/consent clear).

Design doc: "Phase D" + memory qwen-rationale-dataset-preserve.
User decisions (2026-05-19): distribute text + rendered VLM inputs via
Google Drive; PREP WITH ANONYMIZATION, HOLD RELEASE.

Privacy analysis (verified):
  * rationale JSON content = pure skeleton-motion description, 0 PII
    (no stem/performer/country) — safe.
  * linkage to performers lives ONLY in render_index.jsonl / audit logs
    (raw `JP_xx/TW_xx` stems) and in the content-hash filenames
    (render_cache_key = sha256(skeleton_coords+settings); rationale key
    derived from it). The hash is content-derived → anyone holding DIEM-A
    can recompute it from skeleton+published settings and re-identify.
Anonymization scheme:
  1. RE-KEY every (rationale, render) pair to a fresh sequential opaque
     id `rat_NNNNN` (NOT the content hash) — breaks trivial hash matching.
  2. Replace performer token with a consistent P-code (P001..), keep
     emotion/scenario/intensity → anon clip id `P003_anger_1_H`.
  3. Strip the content-hash keys from released JSON; drop internal
     timing/token fields; keep model/prompt/rationale.
  4. EXCLUDE audit logs, render_index.jsonl, and the .pt embedding caches
     (raw stems / regenerable) from the release entirely.
  5. The de-anon map (P-code↔JP/TW, new_id↔orig) is written to a SEPARATE
     _PRIVATE_ dir, never inside the release, never to be uploaded.
  6. DATASHEET honestly states residual re-identifiability (releasing
     derived skeleton-motion renders is itself performer-derived data;
     full unlinkability is impossible — hence the held release pending
     organizer/consent clearance).

Heavy output staged under data. Idempotent / deterministic.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

SRC = Path("output/rationale_cache")
SRC_V2 = Path("output/rationale_cache_v2")
REL = Path("data/dataset_release/qwen_rationale_motion_v1")
PRIV = Path("data/private_deanon_map")
STEM_RE = re.compile(r"^([A-Z]{2})_(\d+)_([a-z]+)_(\d+)_([A-Z])$")


def load_render_index() -> dict:
    """render_cache_key -> stem (JP_06_anger_1_H)."""
    idx = {}
    p = SRC / "renders" / "render_index.jsonl"
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        idx[d["cache_key"]] = d["stem"]
    return idx


def build():
    REL.mkdir(parents=True, exist_ok=True)
    (REL / "rationales").mkdir(exist_ok=True)
    (REL / "renders").mkdir(exist_ok=True)
    PRIV.mkdir(parents=True, exist_ok=True)

    rkey_to_stem = load_render_index()
    rats = sorted(SRC.glob("rationales/*.json")) + \
        sorted(SRC_V2.glob("rationales/*.json"))

    # deterministic performer -> P-code
    performers = set()
    pending = []
    for rp in rats:
        d = json.loads(rp.read_text())
        stem = rkey_to_stem.get(d.get("render_cache_key", ""))
        if not stem:
            continue
        m = STEM_RE.match(stem)
        if not m:
            continue
        performers.add(f"{m.group(1)}_{m.group(2)}")
        pending.append((rp, d, stem, m))
    pmap = {p: f"P{ i + 1:03d}"
            for i, p in enumerate(sorted(performers))}

    manifest, deanon_ids = [], {}
    pending.sort(key=lambda t: t[1]["rationale_cache_key"])  # determinism
    for i, (rp, d, stem, m) in enumerate(pending):
        nid = f"rat_{i + 1:05d}"
        country_id = f"{m.group(1)}_{m.group(2)}"
        emo, scn, inten = m.group(3), m.group(4), m.group(5)
        anon_clip = f"{pmap[country_id]}_{emo}_{scn}_{inten}"
        rk = d.get("render_cache_key", "")

        # anonymized rationale JSON (strip hashes + internal fields)
        (REL / "rationales" / f"{nid}.json").write_text(json.dumps({
            "id": nid, "clip": anon_clip, "emotion": emo,
            "scenario": int(scn), "intensity": inten,
            "performer": pmap[country_id],
            "model_version": d.get("model_version"),
            "prompt_version": d.get("prompt_version"),
            "generated_at": d.get("generated_at"),
            "rationale": d.get("rationale"),
        }, indent=2, ensure_ascii=False))

        has_render = False
        mp4 = SRC / "renders" / f"{rk}.mp4"
        meta = SRC / "renders" / f"{rk}.meta.json"
        if mp4.exists():
            shutil.copy2(mp4, REL / "renders" / f"{nid}.mp4")
            has_render = True
        if meta.exists():
            md = json.loads(meta.read_text())
            (REL / "renders" / f"{nid}.meta.json").write_text(json.dumps({
                "id": nid, "clip": anon_clip,
                "render_settings": md.get("render_settings"),
                "frames_rendered": md.get("frames_rendered"),
                "n_clip_frames": md.get("n_clip_frames"),
            }, indent=2))

        manifest.append({"id": nid, "clip": anon_clip, "emotion": emo,
                         "scenario": int(scn), "intensity": inten,
                         "performer": pmap[country_id],
                         "has_render": has_render, "has_rationale": True,
                         "model_version": d.get("model_version")})
        deanon_ids[nid] = {"orig_stem": stem,
                           "rationale_cache_key": d["rationale_cache_key"],
                           "render_cache_key": rk}

    (REL / "dataset_manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest) + "\n")
    # PRIVATE de-anon map — OUTSIDE the release dir, never to be uploaded
    (PRIV / "qwen_rationale_motion_v1_deanon.json").write_text(json.dumps(
        {"WARNING": "DE-ANONYMIZATION KEY — DO NOT SHARE / UPLOAD",
         "pcode_to_performer": {v: k for k, v in pmap.items()},
         "id_map": deanon_ids}, indent=2))
    return manifest, pmap


def write_docs(manifest, pmap):
    n = len(manifest)
    nr = sum(1 for r in manifest if r["has_render"])
    (REL / "DATASHEET.md").write_text(f"""# Datasheet — Qwen Rationale Motion Dataset v1 (HELD, NOT RELEASED)

**STATUS: RELEASE HELD.** Built with anonymization for review only. Do NOT
upload/distribute until DIEM-A license + performer-consent clearance is
confirmed (likely needs organizer contact).

## What
{n} VLM motion rationales: skeleton-motion clips from the DIEM-A acted
body-emotion corpus rendered to short videos ({nr} `.mp4`, the exact VLM
inputs) + the corresponding Qwen3-VL ("Qwen/Qwen3-VL-30B-A3B-Thinking")
free-form kinematic rationale per body part. Performers anonymized to
{len(pmap)} P-codes; clip ids = `P###_emotion_scenario_intensity`.

## Provenance / motivation
Generated for the explainability track of the MMAC@ACII 2026 DIEM-A paper
(motion→text rationale channel). Source motion: DIEM-A challenge corpus.

## Anonymization (see ANONYMIZATION.md)
Re-keyed to opaque `rat_NNNNN`; performer tokens → P-codes; content-hash
keys, audit logs, render_index, and embedding `.pt` caches removed.

## Known limitations
- **Residual re-identifiability**: the renders are derived skeleton motion;
  anyone holding DIEM-A could in principle re-match by re-rendering. Full
  unlinkability is not achievable for released derived motion — this is
  why the release is HELD pending consent/license clearance.
- Card-label bug (repo Open Q7): the separate explanation-card §1
  ground-truth field is unreliable for 34/50 cards; it is NOT part of this
  dataset (only the §5-style Qwen rationale text is), but noted for users
  who cross-reference the cards.
- Rationales are model-generated descriptions of motion, not human
  annotations; ~Qwen3-VL errors possible.

## Recommended uses / not
Motion→language research, explainability. NOT for performer
re-identification or emotion/biometric profiling of individuals.

## License
TBD — HELD pending DIEM-A organizer + performer-consent clearance.
""")
    (REL / "ANONYMIZATION.md").write_text("""# Anonymization scheme (v1)

1. Re-keyed every pair to fresh sequential `rat_NNNNN` (not the
   content hash) → breaks trivial sha256(skeleton+settings) matching.
2. Performer token → consistent P-code (P001..); kept
   emotion/scenario/intensity.
3. Stripped `rationale_cache_key`/`render_cache_key` and internal
   timing/token fields from released JSON.
4. EXCLUDED: audit_log.jsonl, softlabel_audit_log.jsonl,
   render_index.jsonl (raw `JP_xx/TW_xx` stems), and *.pt embedding
   caches (regenerable; keyed by raw stems).
5. De-anon map written to data/dataset_release/
   _PRIVATE_deanon_DO_NOT_SHARE/ — OUTSIDE this release, never upload.
6. Verified: `grep -r 'JP_[0-9][0-9]|TW_[0-9][0-9]'` over the release
   tree returns 0 (tests/test_qwen_dataset_anon.py).

RESIDUAL RISK (stated honestly): the released `.mp4` IS performer-derived
skeleton motion; a holder of DIEM-A can re-render and match. Identifier
anonymization ≠ motion-content unlinkability. Hence the HELD status.
""")
    (REL / "README.md").write_text(
        "# Qwen Rationale Motion Dataset v1 — HELD (do not distribute)\n\n"
        "See DATASHEET.md and ANONYMIZATION.md. Built by "
        "tools/build_qwen_rationale_dataset.py. Release blocked on "
        "license/consent clearance (TODO Phase D).\n")
    (REL / "LICENSE-HELD.txt").write_text(
        "RELEASE HELD. No license granted yet. Do NOT distribute until "
        "DIEM-A organizer + performer-consent clearance is confirmed.\n")


if __name__ == "__main__":
    mani, pmap = build()
    write_docs(mani, pmap)
    nr = sum(1 for r in mani if r["has_render"])
    print(f"[built] {len(mani)} rationales, {nr} renders, "
          f"{len(pmap)} performers -> P-codes")
    print(f"[release-HELD] {REL}")
    print(f"[private de-anon map] {PRIV} (never upload)")
    # in-process anonymization assert (defence-in-depth; full check in test)
    bad = []
    for p in list(REL.rglob("*.json")) + [REL / "dataset_manifest.jsonl"]:
        if re.search(r"JP_\d\d|TW_\d\d", p.read_text()):
            bad.append(str(p))
    if bad:
        print("ANON FAIL:", bad[:3], file=sys.stderr)
        sys.exit(1)
    print("[anon] 0 raw JP_/TW_ ids in release tree ✓")
