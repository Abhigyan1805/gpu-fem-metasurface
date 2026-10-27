"""Agreement metrics between S11 curves on the common 201-point grid."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

CLIP_DB = -40.0


def notches(f, db, depth_db: float = -6.0, prominence: float = 3.0):
    """Local minima deeper than ``depth_db`` with at least ``prominence`` dB prominence."""
    idx, _ = find_peaks(-np.asarray(db), height=-depth_db, prominence=prominence)
    return [(float(f[i]), float(db[i])) for i in idx]


def match_notches(f, ref_db, test_db):
    """For every reference notch, the nearest test local minimum (prominence >= 1 dB)."""
    cand, _ = find_peaks(-np.asarray(test_db), prominence=1.0)
    out = []
    for fr, dr in notches(f, ref_db):
        if len(cand) == 0:
            out.append(dict(f_ref=fr, depth_ref=dr, f_test=None, depth_test=None, df_pct=None))
            continue
        i = cand[np.argmin(np.abs(f[cand] - fr))]
        out.append(
            dict(
                f_ref=fr,
                depth_ref=dr,
                f_test=float(f[i]),
                depth_test=float(test_db[i]),
                df_pct=100.0 * (float(f[i]) - fr) / fr,
            )
        )
    return out


def compare(f, ref_db, test_db) -> dict:
    ref_c = np.clip(ref_db, CLIP_DB, None)
    test_c = np.clip(test_db, CLIP_DB, None)
    lin = 10 ** (np.asarray(ref_db) / 20) - 10 ** (np.asarray(test_db) / 20)
    m = match_notches(f, ref_db, test_db)
    dfs = [abs(x["df_pct"]) for x in m if x["df_pct"] is not None]
    return dict(
        rmse_db_clip40=float(np.sqrt(np.mean((ref_c - test_c) ** 2))),
        max_abs_db_clip40=float(np.max(np.abs(ref_c - test_c))),
        rmse_lin=float(np.sqrt(np.mean(lin**2))),
        notches=m,
        max_abs_df_pct=float(max(dfs)) if dfs else None,
    )
