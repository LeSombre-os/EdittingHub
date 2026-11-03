#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluation locale synthetique de analyze_audio (alignement/rangs).

Ce script NE modifie pas tests/smoke.py. Il :
  1. genere 3 WAV synthetiques dans outputs/test_wav/ ;
  2. appelle analyze_audio pour chacun ;
  3. calcule et affiche des metriques (tempo, densite, IOI, alignement,
     proportion sur grille, erreur onset-grille, stabilite) ;
  4. chronometre l'analyse et ecrit outputs/beats_eval.json ;
  5. teste l'application d'un offset manuel de 0,25 s.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.audio_beats import analyze_audio  # noqa: E402

OUT_WAV = ROOT / "outputs" / "test_wav"
OUT_JSON = ROOT / "outputs" / "beats_eval.json"
SR = 22050


def _ensure_dirs() -> None:
    OUT_WAV.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)


def _norm(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-9:
        y = y / peak * 0.85
    return y.astype(np.float32)


def make_click_track(path: Path, bpm: float = 120.0, dur: float = 10.0,
                    start: float = 0.1, seed: int = 7) -> list[float]:
    """Piste de clics reguliers (transitoires nets) a bpm BPM."""
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    n = int(round(dur * SR))
    y = np.zeros(n, dtype=np.float64)
    truth: list[float] = []
    t = start
    while t < dur - 0.05:
        truth.append(round(float(t), 6))
        t += period
    for tt in truth:
        idx = int(round(tt * SR))
        L = int(0.020 * SR)
        if idx + L > n:
            continue
        env = np.exp(-np.arange(L) / (0.006 * SR))
        tone = (np.sin(2 * np.pi * 1500.0 * np.arange(L) / SR)
                + 0.35 * (rng.random(L) * 2 - 1)) * env
        y[idx:idx + L] += tone * 0.9
    sf.write(path, _norm(y), SR)
    return truth


def _kick(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    L = int(0.18 * sr)
    if L > n:
        L = n
    freqs = np.linspace(160.0, 42.0, L)
    phase = 2.0 * np.pi * np.cumsum(freqs) / sr
    env = np.exp(-np.arange(L) / (0.035 * sr))
    click = np.sin(phase) * env * 0.95
    sub = np.sin(2 * np.pi * 55.0 * np.arange(L) / sr) * np.exp(-np.arange(L) / (0.06 * sr)) * 0.4
    return (click + sub).astype(np.float64)


def _hat(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    L = int(0.035 * sr)
    if L > n:
        L = n
    noise = rng.random(L) * 2 - 1
    env = np.exp(-np.arange(L) / (0.012 * sr))
    hp = np.diff(noise, prepend=noise[:1])
    return (hp * env * 0.5).astype(np.float64)


def make_groove(path: Path, bpm: float = 100.0, dur: float = 10.0,
                start: float = 0.1, seed: int = 11) -> tuple[list[float], list[float]]:
    """Groove : kick graves (temps), charley (contre-temps), basse tenue + parole."""
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    n = int(round(dur * SR))
    y = np.zeros(n, dtype=np.float64)

    kick_truth: list[float] = []
    t = start
    while t < dur - 0.2:
        kick_truth.append(round(float(t), 6))
        t += period
    hat_truth: list[float] = []
    t = start + period / 2.0
    while t < dur - 0.1:
        hat_truth.append(round(float(t), 6))
        t += period

    for tt in kick_truth:
        idx = int(round(tt * SR))
        k = _kick(n, SR, rng)
        L = k.size
        if idx + L <= n:
            y[idx:idx + L] += k

    for tt in hat_truth:
        idx = int(round(tt * SR))
        h = _hat(n, SR, rng)
        L = h.size
        if idx + L <= n:
            y[idx:idx + L] += h

    bass = 0.22 * np.sin(2 * np.pi * 55.0 * np.arange(n) / SR)
    y += bass

    for a, b in ((2.0, 3.2), (6.5, 7.9)):
        i0 = int(round(a * SR))
        i1 = min(n, int(round(b * SR)))
        if i1 <= i0:
            continue
        seglen = i1 - i0
        carrier = np.sin(2 * np.pi * (280.0 + 120.0 * np.sin(2 * np.pi * 3.0 * np.arange(seglen) / SR))
                           * np.arange(seglen) / SR)
        env_noise = rng.random(seglen)
        w = int(0.05 * SR)
        env = np.convolve(env_noise, np.ones(w) / w, mode="same")
        env = np.clip(env, 0.0, 1.0)
        y[i0:i1] += (carrier * env * 0.35)

    sf.write(path, _norm(y), SR)
    return kick_truth, hat_truth


def make_break_track(path: Path, bpm: float = 120.0, dur: float = 10.0,
                    start: float = 0.1, silence: tuple[float, float] = (4.0, 6.0),
                    seed: int = 19) -> list[float]:
    """Clics reguliers avec un trou de silence de 2 s."""
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    n = int(round(dur * SR))
    y = np.zeros(n, dtype=np.float64)
    truth: list[float] = []
    t = start
    a_s, b_s = silence
    while t < dur - 0.05:
        if not (a_s <= t < b_s):
            truth.append(round(float(t), 6))
        t += period

    for tt in truth:
        idx = int(round(tt * SR))
        L = int(0.020 * SR)
        if idx + L > n:
            continue
        env = np.exp(-np.arange(L) / (0.006 * SR))
        tone = (np.sin(2 * np.pi * 1500.0 * np.arange(L) / SR)
                + 0.35 * (rng.random(L) * 2 - 1)) * env
        y[idx:idx + L] += tone * 0.9

    i0 = int(round(a_s * SR))
    i1 = min(n, int(round(b_s * SR)))
    y[i0:i1] = 0.0

    sf.write(path, _norm(y), SR)
    return truth


def _median_or_none(x) -> float | None:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    v = float(np.median(x))
    if math.isnan(v):
        return None
    return round(v, 6)


def _metrics_for_track(name: str, path: Path, expected_tempo: float,
                       truth: list[float], dur: float) -> dict:
    t0 = time.perf_counter()
    res = analyze_audio(str(path), sr=SR)
    wall = time.perf_counter() - t0

    beats = sorted(res["beats"], key=lambda b: b["time"])
    times = np.array([b["time"] for b in beats], dtype=float)
    duration = float(res["duration"])
    tempo = float(res["tempo"])

    iois = np.diff(times) if times.size >= 2 else np.array([])
    ioi_median = _median_or_none(iois)

    errs_ms: list[float] = []
    matched_15 = 0
    if truth and times.size:
        for t in truth:
            idx = int(np.argmin(np.abs(times - t)))
            e = (times[idx] - t) * 1000.0
            errs_ms.append(float(e))
            if abs(e) < 15.0:
                matched_15 += 1
    errs_arr = np.asarray(errs_ms, dtype=float)
    median_abs_err_ms = (float(np.median(np.abs(errs_arr))) if errs_arr.size else None)
    max_abs_err_ms = (float(np.max(np.abs(errs_arr))) if errs_arr.size else None)

    period = 60.0 / expected_tempo if expected_tempo > 0 else None
    grid_tol = max(0.040, period * 0.08) if period else 0.040
    origin = truth[0] if truth else 0.0
    on_grid = 0
    grid_errs: list[float] = []
    if times.size and period:
        for tt in times:
            k = round((tt - origin) / period)
            nearest = origin + k * period
            d = abs(tt - nearest)
            grid_errs.append(d)
            if d <= grid_tol:
                on_grid += 1
    prop_on_grid = (on_grid / len(times)) if times.size else 0.0
    median_grid_err_ms = (_median_or_none(np.asarray(grid_errs) * 1000.0)
                         if grid_errs else None)

    ranks = res.get("ranks", {}) or {}
    ranks_out = {str(k): int(v) for k, v in ranks.items()}

    out = {
        "name": name,
        "path": str(path),
        "expected_tempo": expected_tempo,
        "ground_truth_count": len(truth),
        "duration_s": round(duration, 3),
        "analysis_elapsed_s": round(float(res["elapsed"]), 3),
        "wall_clock_s": round(wall, 3),
        "tempo_detected": round(tempo, 2),
        "tempo_error": round(tempo - expected_tempo, 2),
        "n_beats": len(beats),
        "ranks": ranks_out,
        "min_time_s": round(float(times.min()), 3) if times.size else None,
        "max_time_s": round(float(times.max()), 3) if times.size else None,
        "ioi_median_s": (round(float(ioi_median), 4) if ioi_median is not None else None),
        "density_beats_per_s": round(len(beats) / duration, 3) if duration > 0 else None,
        "ioi_expected_s": round(period, 4) if period else None,
        "alignment": {
            "matched_within_15ms": matched_15,
            "ground_truth_count": len(truth),
            "coverage_15ms": round(matched_15 / len(truth), 3) if truth else None,
            "median_abs_error_ms": (round(median_abs_err_ms, 2) if median_abs_err_ms is not None else None),
            "max_abs_error_ms": (round(max_abs_err_ms, 2) if max_abs_err_ms is not None else None),
        },
        "grid": {
            "period_s": round(period, 4) if period else None,
            "tolerance_ms": round(grid_tol * 1000.0, 1),
            "proportion_on_grid": round(float(prop_on_grid), 3),
            "on_grid_count": int(on_grid),
            "median_grid_error_ms": median_grid_err_ms,
        },
        "audio_offset_ms": res.get("audio_offset_ms"),
        "manual_offset_s": res.get("manual_offset_s"),
        "total_onsets": int(res.get("total_onsets", 0)),
        "beat_times": [round(float(t), 3) for t in times.tolist()],
    }
    return out, res


def run_stability(path: Path) -> dict:
    r1 = analyze_audio(str(path), sr=SR)
    r2 = analyze_audio(str(path), sr=SR)
    t1 = np.array(sorted(b["time"] for b in r1["beats"]), dtype=float)
    t2 = np.array(sorted(b["time"] for b in r2["beats"]), dtype=float)
    if t1.size and t2.size:
        diffs = []
        used = set()
        for tt in t1:
            if t2.size == 0:
                break
            j = int(np.argmin(np.abs(t2 - tt)))
            if j in used:
                order = np.argsort(np.abs(t2 - tt))
                for cand in order:
                    if cand not in used:
                        j = int(cand)
                        break
            used.add(j)
            diffs.append((t2[j] - tt) * 1000.0)
        diff_arr = np.asarray(diffs, dtype=float)
        median_diff = float(np.median(np.abs(diff_arr))) if diff_arr.size else float("nan")
    else:
        median_diff = float("nan")
    return {
        "run1_beats": len(t1),
        "run2_beats": len(t2),
        "same_count": bool(len(t1) == len(t2)),
        "median_abs_time_diff_ms": (round(median_diff, 2) if not math.isnan(median_diff) else None),
    }


def run_offset_test(path: Path) -> dict:
    base = analyze_audio(str(path), sr=SR, offset=0.0)
    off = analyze_audio(str(path), sr=SR, offset=0.25)
    bt = np.array(sorted(b["time"] for b in base["beats"]), dtype=float)
    ot = np.array(sorted(b["time"] for b in off["beats"]), dtype=float)
    expected = bt + 0.25
    # Appariement par plus proche voisin (l'offset est applique avant
    # classification, donc certaines entrees peuvent etre rejetees).
    matched_shifts = []
    matched = 0
    for oe in ot:
        if expected.size == 0:
            break
        j = int(np.argmin(np.abs(expected - oe)))
        if abs(expected[j] - oe) <= 0.040:
            matched += 1
            matched_shifts.append((oe - expected[j]) * 1000.0)
    if matched_shifts:
        arr = np.asarray(matched_shifts, dtype=float)
        median_shift_ms = float(np.median(np.abs(arr)))
        max_shift_ms = float(np.max(np.abs(arr)))
        all_within = bool(np.all(np.abs(arr) < 20.0))
    else:
        median_shift_ms = float("nan")
        max_shift_ms = float("nan")
        all_within = False
    return {
        "base_beat_count": int(len(bt)),
        "offset_0.25_beat_count": int(len(ot)),
        "same_count": bool(len(bt) == len(ot)),
        "matched_survivors": int(matched),
        "survivor_coverage": round(matched / len(ot), 3) if len(ot) else None,
        "median_shift_ms": (round(median_shift_ms, 2) if not math.isnan(median_shift_ms) else None),
        "max_shift_ms": (round(max_shift_ms, 2) if not math.isnan(max_shift_ms) else None),
        "manual_offset_s_reported": off.get("manual_offset_s"),
        "all_shifted_within_20ms": all_within,
    }


def print_track_summary(m: dict) -> None:
    a = m["alignment"]
    g = m["grid"]
    print("\n-- " + m["name"] + " --")
    print("  duree=%ss  tempo attendu=%s  detecte=%s (err %+0.2f)  elapsed=%ss"
          % (m["duration_s"], m["expected_tempo"], m["tempo_detected"], m["tempo_error"], m["wall_clock_s"]))
    print("  beats=%s  densite=%s/s  rangs=%s  min=%ss max=%ss"
          % (m["n_beats"], m["density_beats_per_s"], m["ranks"], m["min_time_s"], m["max_time_s"]))
    print("  IOI median=%ss (attendu %ss)  onsets totaux=%s"
          % (m["ioi_median_s"], m["ioi_expected_s"], m["total_onsets"]))
    print("  alignement: couverture <=15ms=%s/%s (%s)  mediane err=%sms  max=%sms"
          % (a["matched_within_15ms"], a["ground_truth_count"], a["coverage_15ms"],
             a["median_abs_error_ms"], a["max_abs_error_ms"]))
    print("  grille: on-grid=%s/%s (%s)  tol=%sms  err mediane grille=%sms"
          % (g["on_grid_count"], m["n_beats"], g["proportion_on_grid"],
             g["tolerance_ms"], g["median_grid_error_ms"]))


def main() -> int:
    _ensure_dirs()
    print("EdittingHub - evaluation locale des beats (WAV synthetiques)")

    click_path = OUT_WAV / "click_120bpm_10s.wav"
    groove_path = OUT_WAV / "groove_100bpm_10s.wav"
    break_path = OUT_WAV / "break_silence2s_120bpm.wav"

    click_truth = make_click_track(click_path, bpm=120.0, dur=10.0, start=0.1, seed=7)
    _groove_kicks, groove_truth = make_groove(groove_path, bpm=100.0, dur=10.0, start=0.1, seed=11)
    break_truth = make_break_track(break_path, bpm=120.0, dur=10.0, start=0.1, seed=19)

    print("  genere: %s (%s clics)" % (click_path.name, len(click_truth)))
    print("  genere: %s (%s kicks)" % (groove_path.name, len(groove_truth)))
    print("  genere: %s (%s clics)" % (break_path.name, len(break_truth)))

    results = []
    m_click, _ = _metrics_for_track("click_120bpm", click_path, 120.0, click_truth, 10.0)
    m_groove, _ = _metrics_for_track("groove_100bpm", groove_path, 100.0, groove_truth, 10.0)
    m_break, _ = _metrics_for_track("break_silence2s", break_path, 120.0, break_truth, 10.0)
    results.extend([m_click, m_groove, m_break])

    for m in results:
        print_track_summary(m)

    print("\n-- stabilite (relance click) --")
    stab = run_stability(click_path)
    print("  run1=%s  run2=%s  same_count=%s  diff mediane=%sms"
          % (stab["run1_beats"], stab["run2_beats"], stab["same_count"], stab["median_abs_time_diff_ms"]))
    m_click["stability"] = stab

    print("\n-- test offset 0.25 --")
    offres = run_offset_test(click_path)
    print("  beats base=%s  offsetes=%s  same_count=%s"
          % (offres["base_beat_count"], offres["offset_0.25_beat_count"], offres["same_count"]))
    print("  derivee mediane=%sms  max=%sms  all<20ms=%s  manual_offset_s=%s"
          % (offres["median_shift_ms"], offres["max_shift_ms"], offres["all_shifted_within_20ms"],
             offres["manual_offset_s_reported"]))
    if offres["all_shifted_within_20ms"]:
        print("  OK : offset applique correctement (+0.25 s)")
    else:
        print("  ATTENTION : derivee d'offset inattendue")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rate": SR,
        "tracks": results,
        "offset_test": offres,
        "stability_track": "click_120bpm",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nJSON ecrit : %s" % OUT_JSON)

    print("\n=== RESUME ===")
    for m in results:
        a = m["alignment"]
        g = m["grid"]
        print("  %s: tempo %s/%s  beats %s  couv15ms %s/%s  on-grid %s  IOI %ss  elapsed %ss"
              % (m["name"], m["tempo_detected"], m["expected_tempo"], m["n_beats"],
                 a["matched_within_15ms"], a["ground_truth_count"], g["proportion_on_grid"],
                 m["ioi_median_s"], m["wall_clock_s"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
