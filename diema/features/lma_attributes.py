"""DIEM-A BVH-24 → 32-D Laban Movement Analysis (LMA) attributes (exp087).

Extends exp079's 5 classical kinematics to a 32-D LMA schema
(Body / Effort / Shape / Space × 8 each) so the Bonus-Explainability
narrative connects to body-language research vocabulary (BoLD/ARBEE,
Integrating-LMA arXiv 2023) and feeds the exp089 evidence-grounded narrator.

Schema is fixed in docs/analysis/lma_motion_text_plan.md §1. All features
are clip-level scalars from FK 3D world positions (reuses
multi_stream.compute_joint_positions, like bvh_to_h36m17 / exp079).

Output: (32,) float32 per clip; LMA_NAMES gives the ordered names.

BVH-24 idx: 0 Hips,1 Spine,2 Spine1,3 Spine2,4 Spine3,5 Neck,6 Neck1,7 Head,
 8 RShoulder,9 RArm,10 RForeArm,11 RHand,12 LShoulder,13 LArm,14 LForeArm,
 15 LHand,16 RUpLeg,17 RLeg,18 RFoot,19 RToe,20 LUpLeg,21 LLeg,22 LFoot,23 LToe
"""

from __future__ import annotations

import numpy as np

from diema.features.multi_stream import compute_joint_positions

LMA_NAMES = [
    # Body (8)
    "body.head_bow", "body.trunk_lean", "body.arm_openness",
    "body.lr_asym_speed", "body.lr_asym_pos", "body.stillness_ratio",
    "body.head_lateral_tilt", "body.shoulder_drop",
    # Effort (8)
    "effort.suddenness", "effort.sustainedness", "effort.strong_proxy",
    "effort.light_proxy", "effort.bound_proxy", "effort.free_proxy",
    "effort.intensity_peak", "effort.intensity_var",
    # Shape (8)
    "shape.body_volume", "shape.shoulder_width", "shape.head_height",
    "shape.contraction_change", "shape.rise_sink", "shape.spread_change",
    "shape.enclose_proxy", "shape.advance_recede",
    # Space (8)
    "space.directness", "space.path_curvature", "space.root_xy_disp",
    "space.root_z_disp", "space.root_sway_xy_rms", "space.cumulative_turn",
    "space.dominant_direction", "space.locomotion_ratio",
]
assert len(LMA_NAMES) == 32

# BVH joint indices
HIPS, SPINE3, NECK, HEAD = 0, 4, 5, 7
R_SHO, R_ELB, R_WRI = 9, 10, 11
L_SHO, L_ELB, L_WRI = 13, 14, 15
R_HIP, R_KNE, R_ANK = 16, 17, 18
L_HIP, L_KNE, L_ANK = 20, 21, 22
# left/right paired joints for asymmetry (BVH lr_pairs)
LR_PAIRS = [(12, 8), (13, 9), (14, 10), (15, 11), (20, 16), (21, 17), (22, 18), (23, 19)]
_EPS = 1e-8


def _safe(v: float) -> float:
    return float(v) if np.isfinite(v) else 0.0


def compute_lma_attributes(root_pos: np.ndarray, joint_quats: np.ndarray,
                            offsets: np.ndarray | None = None) -> np.ndarray:
    """One DIEM-A clip → (32,) float32 LMA attribute vector.

    Parameters
    ----------
    root_pos : (F, 3) world-frame root translation.
    joint_quats : (F, 24, 4) per-frame quaternion (wxyz) joint rotations.
    offsets : (24, 3) optional per-joint rest-pose offsets. If None, uses
        the hardcoded JP_06 reference (back-compat). Pass custom offsets
        to compute LMA on retargeted clips (exp099 bone-retarget pipeline).
    """
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must be (F,3), got {root_pos.shape}")
    if joint_quats.ndim != 3 or joint_quats.shape[1:] != (24, 4):
        raise ValueError(f"joint_quats must be (F,24,4), got {joint_quats.shape}")
    if root_pos.shape[0] != joint_quats.shape[0]:
        raise ValueError("frame count mismatch")

    P = compute_joint_positions(root_pos, joint_quats,
                                  offsets=offsets).astype(np.float64)  # (F,24,3)
    F = P.shape[0]
    V = np.diff(P, axis=0) if F > 1 else np.zeros((1, 24, 3))   # (F-1,24,3)
    A = np.diff(V, axis=0) if V.shape[0] > 1 else np.zeros((1, 24, 3))
    J = np.diff(A, axis=0) if A.shape[0] > 1 else np.zeros((1, 24, 3))
    spd = np.linalg.norm(V, axis=-1)            # (F-1,24)
    gspeed = spd.mean(axis=1)                   # (F-1,) global mean speed/frame
    up = np.array([0.0, 0.0, 1.0])              # world up = +Z (FK convention)

    pelvis = P[:, HIPS, :]
    head = P[:, HEAD, :]
    sho_mid = 0.5 * (P[:, L_SHO, :] + P[:, R_SHO, :])
    hip_mid = 0.5 * (P[:, L_HIP, :] + P[:, R_HIP, :])

    def _angle_to_up(vec):
        n = np.linalg.norm(vec, axis=-1)
        cos = np.clip((vec @ up) / (n + _EPS), -1, 1)
        return np.arccos(cos)  # (F,)

    # ---- Body ----
    head_bow = _angle_to_up(head - pelvis).mean()
    trunk_lean = _angle_to_up(sho_mid - hip_mid).mean()
    # arm openness: shoulder-elbow-wrist apex angle, avg L/R, avg t
    def _apex(a, b, c):
        u = P[:, a, :] - P[:, b, :]
        w = P[:, c, :] - P[:, b, :]
        cos = np.clip((u * w).sum(-1) /
                      (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1) + _EPS), -1, 1)
        return np.arccos(cos)
    arm_openness = 0.5 * (_apex(L_SHO, L_ELB, L_WRI).mean() +
                          _apex(R_SHO, R_ELB, R_WRI).mean())
    li = [p[0] for p in LR_PAIRS]
    ri = [p[1] for p in LR_PAIRS]
    lr_asym_speed = np.abs(spd[:, li] - spd[:, ri]).mean() if spd.shape[0] else 0.0
    relx = P - pelvis[:, None, :]
    lr_asym_pos = np.abs(np.linalg.norm(relx[:, li], axis=-1) -
                         np.linalg.norm(relx[:, ri], axis=-1)).mean()
    thr = np.median(gspeed) * 0.5 if gspeed.size else 0.0
    stillness_ratio = float((gspeed < thr).mean()) if gspeed.size else 0.0
    head_lateral_tilt = np.abs((head - P[:, NECK, :])[:, 0]).mean()
    shoulder_drop = (P[:, L_SHO, 2] - P[:, R_SHO, 2]).mean()

    # ---- Effort ----
    jmag = np.linalg.norm(J, axis=-1).mean(axis=1) if J.shape[0] else np.zeros(1)
    suddenness = (np.percentile(jmag, 95) / (np.median(jmag) + _EPS)) if jmag.size else 0.0
    if gspeed.size > 8:
        g = gspeed - gspeed.mean()
        sustainedness = (g[:-8] @ g[8:]) / ((g @ g) + _EPS)
    else:
        sustainedness = 0.0
    ke = gspeed ** 2
    strong_proxy = ke.max() if ke.size else 0.0
    light_proxy = ke[ke > ke.mean()].min() if (ke.size and (ke > ke.mean()).any()) else 0.0
    amag = np.linalg.norm(A, axis=-1).mean(axis=1) if A.shape[0] else np.zeros(1)
    # bound: low directional variance of acceleration
    if A.shape[0] > 1:
        an = A.mean(axis=1)
        ad = an / (np.linalg.norm(an, axis=-1, keepdims=True) + _EPS)
        bound_proxy = ad.std()
        vn = V.mean(axis=1)[:-1]  # align to an: V is (F-1), A is (F-2)
        va_cos = np.clip((vn * an).sum(-1) /
                         (np.linalg.norm(vn, axis=-1) *
                          np.linalg.norm(an, axis=-1) + _EPS), -1, 1)
        free_proxy = np.arccos(va_cos).mean()
    else:
        bound_proxy = free_proxy = 0.0
    intensity_peak = gspeed.max() if gspeed.size else 0.0
    intensity_var = gspeed.var() if gspeed.size else 0.0

    # ---- Shape ----
    span = P.max(axis=1) - P.min(axis=1)            # (F,3)
    body_volume = (span[:, 0] * span[:, 1] * span[:, 2]).mean()
    shoulder_width = np.linalg.norm(P[:, L_SHO, :] - P[:, R_SHO, :], axis=-1).mean()
    head_height = (head[:, 2] - pelvis[:, 2]).mean()
    vol_t = span[:, 0] * span[:, 1] * span[:, 2]
    q = max(F // 4, 1)
    contraction_change = ((vol_t[:q].mean() - vol_t[-q:].mean()) /
                          (vol_t.mean() + _EPS))
    rise_sink = head[:, 2].max() - head[:, 2].min()
    sw = np.linalg.norm(P[:, L_SHO, :] - P[:, R_SHO, :], axis=-1)
    spread_change = (sw[-q:].mean() - sw[:q].mean()) / (sw[:q].mean() + _EPS)
    # enclose: wrists toward body midline (small = enclosed)
    midline = 0.5 * (P[:, L_HIP, :] + P[:, R_HIP, :])
    enclose_proxy = 0.5 * (
        np.linalg.norm(P[:, L_WRI, :] - midline, axis=-1).mean() +
        np.linalg.norm(P[:, R_WRI, :] - midline, axis=-1).mean())
    advance_recede = pelvis[-1, 1] - pelvis[0, 1]

    # ---- Space ----
    root_path = np.linalg.norm(np.diff(pelvis, axis=0), axis=-1).sum() if F > 1 else 0.0
    net_disp = np.linalg.norm(pelvis[-1] - pelvis[0])
    directness = net_disp / (root_path + _EPS)
    if F > 2:
        rv = np.diff(pelvis, axis=0)
        rvn = rv / (np.linalg.norm(rv, axis=-1, keepdims=True) + _EPS)
        ccos = np.clip((rvn[:-1] * rvn[1:]).sum(-1), -1, 1)
        path_curvature = np.arccos(ccos).mean()
        cumulative_turn = np.arccos(ccos).sum()
    else:
        path_curvature = cumulative_turn = 0.0
    root_xy_disp = np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2]) if F else 0.0
    root_z_disp = abs(pelvis[-1, 2] - pelvis[0, 2]) if F else 0.0
    if F > 1:
        a0, a1 = pelvis[0, :2], pelvis[-1, :2]
        seg = a1 - a0
        L = np.linalg.norm(seg) + _EPS
        rel = pelvis[:, :2] - a0
        # 2D cross magnitude (scalar) = perpendicular distance × L
        cross2d = seg[0] * rel[:, 1] - seg[1] * rel[:, 0]
        dev = np.abs(cross2d) / L
        root_sway_xy_rms = np.sqrt((dev ** 2).mean())
    else:
        root_sway_xy_rms = 0.0
    if F > 1:
        tot = pelvis[-1, :2] - pelvis[0, :2]
        dominant_direction = np.arctan2(tot[1], tot[0])
        rh = np.linalg.norm(np.diff(pelvis[:, :2], axis=0), axis=-1)
        locomotion_ratio = float((rh > (np.median(rh) + _EPS)).mean())
    else:
        dominant_direction = locomotion_ratio = 0.0

    feats = np.array([
        head_bow, trunk_lean, arm_openness, lr_asym_speed, lr_asym_pos,
        stillness_ratio, head_lateral_tilt, shoulder_drop,
        suddenness, sustainedness, strong_proxy, light_proxy,
        bound_proxy, free_proxy, intensity_peak, intensity_var,
        body_volume, shoulder_width, head_height, contraction_change,
        rise_sink, spread_change, enclose_proxy, advance_recede,
        directness, path_curvature, root_xy_disp, root_z_disp,
        root_sway_xy_rms, cumulative_turn, dominant_direction, locomotion_ratio,
    ], dtype=np.float64)
    return np.array([_safe(x) for x in feats], dtype=np.float32)


def compute_batch(root_pos_list, joint_quats_list) -> np.ndarray:
    if len(root_pos_list) != len(joint_quats_list):
        raise ValueError("length mismatch")
    out = np.zeros((len(root_pos_list), 32), dtype=np.float32)
    for i, (rp, jq) in enumerate(zip(root_pos_list, joint_quats_list)):
        out[i] = compute_lma_attributes(rp, jq)
    return out
