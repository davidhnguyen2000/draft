"""How discrete, how regular and how symmetric a converged gait is, from foot contacts.

  discrete   ``duty_factor``, ``contacts_per_stride``
  regular    ``stride_period_cv``, ``phase_lock``
  symmetric  ``lr_phase_asymmetry`` (timing); :func:`joint_symmetry_metrics` (amplitude)

All metrics are scale-free (phases, ratios, CVs), so designs of different size
compare directly. ``contact`` is ``[T, B, F]`` boolean, feet ordered as ``feet``,
sampled every ``dt`` s; foot 0 is the phase reference. A metric that cannot be
computed for a replica is NaN and dropped from the mean, not counted as zero.
"""
from __future__ import annotations

import numpy as np

#: Canonical footfall patterns: phase of each foot relative to the left-front, in
#: cycles. ``gait_class_dist`` says how close the nearest match actually is.
CANONICAL_GAITS: dict[str, tuple[float, float, float]] = {
    #        (fr,   rl,   rr)   relative to fl
    "trot":  (0.5,  0.5,  0.0),   # diagonal pairs together
    "pace":  (0.5,  0.0,  0.5),   # lateral pairs together
    "bound": (0.0,  0.5,  0.5),   # front pair together, then rear pair
    "pronk": (0.0,  0.0,  0.0),   # all four together
    # A four-beat walk has two mirror-image sequences; both are walks.
    "walk":  (0.5,  0.25, 0.75),  # lateral-sequence four-beat
    "walk*": (0.5,  0.75, 0.25),  # its mirror image
}

DEFAULT_FEET = ("fl", "fr", "rl", "rr")


# ── circular statistics ──────────────────────────────────────────────────────

def _circ_mean_R(phases: np.ndarray) -> tuple[float, float]:
    """Circular mean and resultant length of phases given in CYCLES (0-1).

    ``R`` is the concentration: 1.0 all at one point, 0.0 uniformly spread.
    """
    if phases.size == 0:
        return float("nan"), float("nan")
    z = np.exp(2j * np.pi * phases)
    m = z.mean()
    return float(np.angle(m) / (2 * np.pi) % 1.0), float(np.abs(m))


def _circ_dist(a: float, b: float) -> float:
    """Distance between two phases in cycles, wrapped into [0, 0.5]."""
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


# ── event extraction ─────────────────────────────────────────────────────────

def _debounce(c: np.ndarray, min_run: int) -> np.ndarray:
    """Erase contact and air phases shorter than ``min_run`` samples.

    Touchdown bounce would otherwise be counted as extra footfalls and corrupt
    every event-based metric.
    """
    if min_run <= 1 or c.size == 0:
        return c
    out = c.copy()
    # Run-length encode, then absorb every short run into its predecessor.
    edges = np.flatnonzero(np.diff(out)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [out.size]))
    for i in range(1, len(starts)):                 # never flip the first run
        if ends[i] - starts[i] < min_run:
            out[starts[i]:ends[i]] = out[starts[i] - 1]
    return out


def _touchdowns(c: np.ndarray, dt: float) -> np.ndarray:
    """Times of every rising edge (swing -> stance) in a 1-D contact signal.

    Phases read off these are quantised to ``dt / stride_period`` cycles
    (about 0.02-0.04 at 50 Hz); smaller differences are not meaningful.
    """
    return (np.flatnonzero((~c[:-1]) & c[1:]) + 1).astype(np.float64) * dt


def _per_env(contact: np.ndarray, dt: float,
             min_run_s: float = 0.03) -> dict[str, np.ndarray]:
    """Reduce ``[T, B, F]`` contact to one row of statistics per replica."""
    T, B, F = contact.shape
    duration = T * dt
    nan = float("nan")
    min_run = max(1, int(round(min_run_s / max(dt, 1e-9))))
    contact = np.stack([[_debounce(contact[:, b, f], min_run) for f in range(F)]
                        for b in range(B)], axis=0).transpose(2, 0, 1)

    duty = contact.mean(axis=0)                       # [B, F]
    out = {k: np.full(B, nan) for k in
           ("duty_factor", "duty_spread", "contacts_per_stride",
            "stride_period_s", "stride_period_cv", "phase_lock",
            "lr_phase_asymmetry", "gait_class_dist")}
    out["gait_class"] = np.full(B, "", dtype=object)
    # Per-replica phase of each foot.
    out["phases"] = np.full((B, F), nan)

    for b in range(B):
        # Duty factor needs no period, so write it before any early exit.
        out["duty_factor"][b] = float(duty[b].mean())
        out["duty_spread"][b] = float(duty[b].max() - duty[b].min())

        tds = [_touchdowns(contact[:, b, f], dt) for f in range(F)]
        # A stride needs two touchdowns to have a duration at all.
        intervals = [np.diff(t) for t in tds if t.size >= 2]
        if not intervals:
            continue
        pooled = np.concatenate(intervals)
        # Median: a missed step (double-length interval) should not shift the period.
        period = float(np.median(pooled))
        if not np.isfinite(period) or period <= 0:
            continue
        out["stride_period_s"][b] = period
        out["stride_period_cv"][b] = float(pooled.std() / max(pooled.mean(), 1e-9))
        # Touchdowns per foot per stride: 1.0 is clean, >1 is chattering/scuffing.
        n_strides = duration / period
        out["contacts_per_stride"][b] = float(
            np.mean([t.size for t in tds]) / max(n_strides, 1e-9))

        # Phase of every foot against the reference foot's own cycle.
        ref = tds[0]
        if ref.size < 2:
            continue
        phi, R = np.full(F, np.nan), np.full(F, np.nan)
        phi[0], R[0] = 0.0, 1.0
        for f in range(1, F):
            if tds[f].size == 0:
                continue
            # Fraction of the enclosing reference cycle; touchdowns before the
            # first reference touchdown are dropped.
            idx = np.searchsorted(ref, tds[f], side="right") - 1
            ok = idx >= 0
            if not ok.any():
                continue
            ph = ((tds[f][ok] - ref[idx[ok]]) / period) % 1.0
            phi[f], R[f] = _circ_mean_R(ph)
        out["phases"][b] = phi
        # How reliably the non-reference feet hit their phase slot.
        out["phase_lock"][b] = float(np.nanmean(R[1:])) if F > 1 else nan

        if F >= 4 and np.isfinite(phi[1:4]).all():
            # Mirror symmetry for any gait: the front pair's L->R offset must
            # equal the rear pair's.
            front = phi[1] - phi[0]          # fr - fl
            rear = phi[3] - phi[2]           # rr - rl
            # Distance in cycles: 0 perfect, 0.5 worst.
            out["lr_phase_asymmetry"][b] = _circ_dist(front, rear)

            name, dist = classify_gait(tuple(phi[1:4]))
            out["gait_class"][b] = name
            out["gait_class_dist"][b] = dist
    return out


def classify_gait(phases: tuple[float, float, float]) -> tuple[str, float]:
    """Nearest canonical footfall pattern, and the phase distance to it.

    Distance is the RMS circular error over the non-reference feet, in cycles.
    """
    best, best_d = "", float("inf")
    for name, ref in CANONICAL_GAITS.items():
        d = float(np.sqrt(np.mean([_circ_dist(p, r) ** 2
                                   for p, r in zip(phases, ref)])))
        if d < best_d:
            best, best_d = name, d
    return best, best_d


# ── public entry points ──────────────────────────────────────────────────────

def contact_gait_metrics(contact, dt: float,
                         feet: tuple[str, ...] = DEFAULT_FEET) -> dict:
    """Gait discreteness, regularity and timing symmetry from foot contacts.

    ``contact`` is ``[T, B, F]``, boolean or a count (>0 means touching), as a
    torch tensor or numpy array. Returns plain floats averaged over the replicas
    that produced a usable answer, plus ``gait_class`` as the modal name.
    """
    c = np.asarray(contact.detach().cpu() if hasattr(contact, "detach")
                   else contact)
    if c.ndim != 3 or c.shape[0] < 3:
        return {}
    c = c > 0
    st = _per_env(c, dt)

    def avg(k):
        v = st[k]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    names = [n for n in st["gait_class"] if n]
    modal, share = "", float("nan")
    if names:
        vals, counts = np.unique(np.asarray(names), return_counts=True)
        modal = str(vals[counts.argmax()])
        # Fraction of replicas agreeing on the modal gait.
        share = float(counts.max() / len(names))

    # Fraction of replicas with a measurable stride; every metric below is
    # conditioned on it, so read it first.
    defined = float(np.isfinite(st["stride_period_s"]).mean())
    out = {
        "gait_defined_frac": defined,
        "duty_factor": avg("duty_factor"),
        "duty_spread": avg("duty_spread"),
        "contacts_per_stride": avg("contacts_per_stride"),
        "stride_period_s": avg("stride_period_s"),
        "stride_period_cv": avg("stride_period_cv"),
        "gait_phase_lock": avg("phase_lock"),
        "gait_lr_phase_asymmetry": avg("lr_phase_asymmetry"),
        "gait_class": modal,
        "gait_class_share": share,
        "gait_class_dist": avg("gait_class_dist"),
    }
    # Mean phase of each non-reference foot.
    ph = st["phases"]
    for f in range(1, min(len(feet), ph.shape[1])):
        col = ph[:, f]
        col = col[np.isfinite(col)]
        out[f"phase_{feet[f]}"] = _circ_mean_R(col)[0] if col.size else float("nan")
    return out


def joint_symmetry_metrics(joint_pos, joint_names: list[str],
                           pairs=(("fl", "fr"), ("rl", "rr"))) -> dict:
    """Left-right symmetry of how FAR the legs swing, from joint positions.

    ``joint_pos`` is ``[T, B, J]``; ``fl_x`` pairs with ``fr_x`` (etc.). Returns the
    symmetry index ``|exc_L - exc_R| / mean(exc)``: 0 is perfect.
    """
    q = np.asarray(joint_pos.detach().cpu() if hasattr(joint_pos, "detach")
                   else joint_pos)
    if q.ndim != 3 or q.shape[0] < 3:
        return {}
    # 5th-95th percentile swing, robust to a single stumble.
    exc = np.percentile(q, 95, axis=0) - np.percentile(q, 5, axis=0)   # [B, J]
    by_name = {n: i for i, n in enumerate(joint_names)}

    idx: list[float] = []
    for left, right in pairs:
        for name, i in by_name.items():
            if not name.startswith(left + "_"):
                continue
            j = by_name.get(right + "_" + name[len(left) + 1:])
            if j is None:
                continue
            a, b = exc[:, i], exc[:, j]
            denom = 0.5 * (a + b)
            good = denom > 1e-6
            if good.any():
                idx.append(float(np.mean(np.abs(a[good] - b[good]) / denom[good])))
    if not idx:
        return {}
    return {"gait_lr_excursion_symmetry_index": float(np.mean(idx))}
