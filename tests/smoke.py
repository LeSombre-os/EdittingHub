#!/usr/bin/env python3
"""Tests de fumée EdittingHub — aucune dépendance externe, aucun réseau.

    python3 tests/smoke.py            # tests unitaires (rapides, < 1 s)
    python3 tests/smoke.py --api      # + tests HTTP sur le serveur 8005

Couvre les régressions historiques :
  - marqueurs FCPXML (structure asset-clip, nombre, références)
  - script DaVinci : compilation, aucune ligne indentée (collage console),
    frames entiers, ordre croissant, filtrage des rangs
  - installation dans le dossier Scripts de Resolve (dans un dossier temporaire)
  - exports CSV / EDL / FCPXML / davinci_script
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from validate_fcpxml import validate_file as _validate_fcpxml_file
except Exception:
    _validate_fcpxml_file = None

from app.davinci import build_script, install_script          # noqa: E402
from app.fcpxml import markers_document, cuts_document          # noqa: E402
from app.services.audio_beats import export_markers            # noqa: E402
from app.services.rhythm import (build_cut_plan, cut_points,   # noqa: E402
                                 export_rhythm)

BEATS = [
    {"time": 0.116, "rank": 1, "label": "Impact fort", "energy": 13.7},
    {"time": 0.418, "rank": 2, "label": "Intermédiaire", "energy": 3.3},
    {"time": 0.604, "rank": 1, "label": "Impact fort", "energy": 9.1},
    {"time": 1.463, "rank": 2, "label": "Intermédiaire", "energy": 4.8},
    {"time": 1.672, "rank": 3, "label": "Micro-variation", "energy": 1.2},
    {"time": 2.000, "rank": 1, "label": "Impact fort", "energy": 11.0},
]

RHYTHM_BEATS = [
    {"time": 0.5, "rank": 1}, {"time": 1.0, "rank": 2},
    {"time": 1.3, "rank": 1}, {"time": 2.5, "rank": 1},
    {"time": 2.9, "rank": 1}, {"time": 5.0, "rank": 2},
    {"time": 7.2, "rank": 1}, {"time": 7.6, "rank": 1},
]

_results: list[tuple[bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    _results.append((bool(condition), label))
    mark = "OK  " if condition else "FAIL"
    print(f"  {mark} {label}" + (f"  ({detail})" if detail else ""))


# ── Faux Resolve pour exécuter le script généré sans DaVinci ───────────────

class _FakeTimeline:
    def __init__(self) -> None:
        self.markers: list[tuple[int, str, str]] = []
        self.name = "Timeline de test"

    def GetName(self) -> str:
        return self.name

    def GetSetting(self, key: str):
        return "24" if key == "timelineFrameRate" else None

    def AddMarker(self, frame, color, name, note, duration, custom=None):
        if not isinstance(frame, int) or isinstance(frame, bool):
            raise TypeError(f"frame doit être un int, reçu {type(frame).__name__}")
        self.markers.append((frame, color, name))
        return True

    def DeleteMarkersByColor(self, _color) -> None:
        pass


class _FakeResolve:
    def __init__(self, timeline):
        self._tl = timeline

    def GetProjectManager(self):
        tl = self._tl

        class _PM:
            def GetCurrentProject(self):
                class _Project:
                    def GetCurrentTimeline(self):
                        return tl
                return _Project()
        return _PM()


def run_script(text: str, timeline=None):
    """Exécute le script généré avec un faux Resolve et retourne la timeline."""
    tl = timeline or _FakeTimeline()
    scope = {"resolve": _FakeResolve(tl), "__name__": "__main__"}
    buffer = io.StringIO()
    old_stdout, sys.stdout = sys.stdout, buffer
    try:
        exec(compile(text, "<script davinci>", "exec"), scope)
    finally:
        sys.stdout = old_stdout
    return tl, buffer.getvalue()


# ── Tests ──────────────────────────────────────────────────────────────────

def test_script() -> None:
    print("\n[1] Script DaVinci")
    script = build_script(BEATS, fps=24, title="demo", asset_path="/tmp/demo.mp4",
                          ranks=[1, 2, 3])
    try:
        compile(script, "<gen>", "exec")
        check("le script compile", True)
    except SyntaxError as exc:
        check("le script compile", False, str(exc))
        return

    indented = [l for l in script.splitlines()
                if l[:1] in (" ", "\t") and l.strip() and not l.lstrip().startswith("#")]
    check("aucune ligne indentée (collage console)", not indented,
          f"{len(indented)} ligne(s)")

    tl, out = run_script(script)
    check("6 marqueurs posés", len(tl.markers) == 6, str(len(tl.markers)))
    check("frames entiers", all(isinstance(m[0], int) for m in tl.markers))
    check("frames croissants",
          [m[0] for m in tl.markers] == sorted(m[0] for m in tl.markers))
    check("frame du 1er beat = 3", tl.markers and tl.markers[0][0] == 3,
          str(tl.markers[0][0] if tl.markers else None))
    check("sortie ASCII", all(ord(c) < 128 for c in out))
    check("message final", "OK - marqueurs ajoutes" in out)

    filtre = build_script(BEATS, fps=24, ranks=[1])
    tl1, _ = run_script(filtre)
    check("filtre rang 1 -> 3 marqueurs", len(tl1.markers) == 3, str(len(tl1.markers)))
    check("couleur rouge pour rang 1",
          all(m[1] == "Red" for m in tl1.markers))

    vide = build_script([], fps=24, ranks=None)
    tl0, out0 = run_script(vide)
    check("liste vide -> 0 marqueur", len(tl0.markers) == 0)
    check("liste vide -> pas de crash", "0 / 0" in out0)

    hostile = build_script([{"time": 1.0, "rank": 1, "label": 'a"b|c;d\\e'}],
                           fps=24, ranks=[1])
    try:
        compile(hostile, "<hostile>", "exec")
        tlh, _ = run_script(hostile)
        check("libellé hostile neutralisé", len(tlh.markers) == 1)
    except SyntaxError as exc:
        check("libellé hostile neutralisé", False, str(exc))


def test_fcpxml() -> None:
    print("\n[2] FCPXML")
    markers = [{"time": b["time"], "value": f"Beat {i}",
                "note": f"Rang {b['rank']}"} for i, b in enumerate(BEATS, 1)]
    doc = markers_document("/tmp/demo.mp4", 24, markers, title="demo",
                           duration=3.0)
    try:
        root = ET.fromstring(doc.split("\n", 2)[2])
        check("XML bien formé", True)
    except ET.ParseError as exc:
        check("XML bien formé", False, str(exc))
        return

    found = root.findall(".//marker")
    check("6 markers", len(found) == 6, str(len(found)))
    check("markers sous asset-clip",
          root.find(".//asset-clip/marker") is not None)
    check("asset avec media-rep", root.find(".//asset/media-rep") is not None)
    check("sequence avec spine", root.find(".//sequence/spine") is not None)


def test_fcpxml_scene_valid() -> None:
    print("\\n[2b] FCPXML scènes (dict précalculé)")
    video = "/media/lesombre/2A14-B525/SSD_E/Opencode/_externes/test_vidéo/fragment-trailer.mp4"
    if not os.path.isfile(video):
        check("vidéo de fixture présente", False, video)
        return
    # Dict précalculé : inclut un plan nul et un plan hors bornes, qui doivent
    # être ignorés par cuts_document (anti-flaky : pas de scenedetect réel).
    cuts = [
        {"start": 0.0, "duration": 2.0},
        {"start": 2.0, "duration": 0.0},
        {"start": 100.0, "duration": 1.0},
        {"start": 4.0, "duration": 1.0 / 30.0},
    ]
    xml = cuts_document(video, 30.0, cuts, title="scene_valid")
    tmp = None
    try:
        try:
            root = ET.fromstring(xml.split("\n", 2)[2])
            check("FCPXML scène : XML bien formé", True)
        except ET.ParseError as exc:
            check("FCPXML scène : XML bien formé", False, str(exc))
            return

        fd = tempfile.NamedTemporaryFile("w", suffix=".fcpxml",
                                         delete=False, encoding="utf-8")
        fd.write(xml)
        fd.close()
        tmp = fd.name
        if _validate_fcpxml_file is not None:
            errs = _validate_fcpxml_file(tmp)
            check("FCPXML scène : validateur vert", not errs,
                  "; ".join(errs)[:120])
        else:
            check("validate_fcpxml importable", False, "module manquant")

        clips = root.findall(".//sequence/spine/asset-clip")
        check("FCPXML scène : 2 clips valides",
              len(clips) == 2, str(len(clips)))
        durs = [ET.fromstring("<x/>") is not None or True for _ in clips]
        # aucune durée < 1 frame
        from fractions import Fraction
        frame = Fraction(1, 30)
        ok_min = all(_parse_rational_safe(c.get("duration")) >= frame
                     for c in clips)
        check("FCPXML scène : durées >= 1 frame", ok_min)
        # sequence duration = somme
        seq = root.find(".//sequence")
        total = sum((_parse_rational_safe(c.get("duration")) for c in clips),
                    Fraction(0, 1))
        check("FCPXML scène : durée séquence = somme",
              _parse_rational_safe(seq.get("duration")) == total,
              f"{seq.get('duration')} vs {total}")
        # src existe
        srcs = [a.find("media-rep").get("src")
                for a in root.findall(".//asset")
                if a.find("media-rep") is not None]
        src_paths = []
        for s in srcs:
            if not s or not s.startswith("file://"):
                src_paths.append(None)
                continue
            fp = unquote(s[len("file://"):])
            if fp.startswith("//"):
                fp = fp[1:]
            src_paths.append(fp)
        exists = all(p is not None and os.path.exists(p) for p in src_paths)
        check("FCPXML scène : src existe", exists)
        # noms sans newline
        names_ok = all("\\n" not in (c.get("name") or "")
                       for c in clips)
        check("FCPXML scène : noms sans newline", names_ok)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _parse_rational_safe(value):
    try:
        from fractions import Fraction
        text = str(value).strip()
        if not text.endswith("s"):
            return Fraction(0)
        core = text[:-1]
        if "/" in core:
            a, b = core.split("/", 1)
            return Fraction(int(a), int(b))
        return Fraction(int(core), 1)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def test_exports() -> None:
    print("\n[3] Exports")
    for fmt in ("csv", "edl", "fcpxml", "davinci_script"):
        text = export_markers(BEATS, fps=24, fmt=fmt, title="demo",
                              asset_path="/tmp/demo.mp4", ranks=[1, 2, 3])
        check(f"{fmt} non vide", bool(text.strip()))

    filtre = export_markers(BEATS, fps=24, fmt="csv", title="demo",
                            asset_path="/tmp/demo.mp4", ranks=[1])
    lignes = [l for l in filtre.splitlines() if l.strip()]
    check("CSV filtré rang 1 -> 4 lignes (en-tête + 3)", len(lignes) == 4,
          str(len(lignes)))

    try:
        export_markers(BEATS, fps=24, fmt="inconnu")
        check("format inconnu -> erreur", False)
    except ValueError:
        check("format inconnu -> erreur", True)


def test_install() -> None:
    print("\n[4] Installation dans Resolve")
    script = build_script(BEATS, fps=24, title="demo", ranks=[1, 2, 3])
    old_home = os.environ.get("HOME")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["HOME"] = tmp
        try:
            info = install_script(script)
            path = Path(info["path"])
            check("fichier écrit", path.is_file(), info["path"])
            check("nom fixe", path.name == "EdittingHub_Beat_Markers.py", path.name)
            check("exécutable", os.access(path, os.X_OK))
            check("contenu identique", path.read_text(encoding="utf-8") == script)
            # Le redémarrage n'est nécessaire que si Resolve tournait déjà au
            # moment de la création du fichier (il n'énumère le dossier qu'au
            # démarrage). Resolve fermé = il le verra au prochain lancement.
            running = bool(info.get("resolve_running"))
            check("1re installation -> restart seulement si Resolve tourne",
                  info["restart_needed"] is running,
                  f"resolve_running={running}")
            info2 = install_script(script)
            check("réinstallation -> pas de restart",
                  info2["restart_needed"] is False)
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home


def test_rhythm() -> None:
    print("\n[5] Montage au rythme")
    points, used = cut_points(RHYTHM_BEATS, 10.0, ranks=[1],
                              min_shot=0.8, max_shot=3.0)
    check("points commencent à 0", points[0] == 0.0, str(points[0]))
    check("points finissent à la durée musique",
          abs(points[-1] - 10.0) < 1e-6, str(points[-1]))
    check("3 beats rang 1 retenus", used == 3, str(used))
    segments = [b - a for a, b in zip(points, points[1:])]
    check("aucun plan < durée min",
          all(s >= 0.8 - 1e-6 for s in segments),
          f"min={min(segments):.3f}")
    check("aucun plan > durée max",
          all(s <= 3.0 + 1e-6 for s in segments),
          f"max={max(segments):.3f}")

    plan = build_cut_plan(RHYTHM_BEATS, video_duration=6.0,
                          audio_duration=10.0, ranks=[1], min_shot=0.8,
                          max_shot=3.0, loop=True, fps=24)
    cuts = plan["cuts"]
    check("plan non vide", len(cuts) > 0, str(len(cuts)))
    check("timeline = durée musique",
          abs(plan["timeline_duration"] - 10.0) < 0.05,
          str(plan["timeline_duration"]))
    check("vidéo bouclée -> plusieurs clips par plan",
          plan["clips"] > plan["segments"],
          f"{plan['clips']} clips / {plan['segments']} plans")
    check("in-points dans la vidéo",
          all(0.0 <= c["start"] < 6.0 for c in cuts))
    check("timecodes source présents",
          all(c["start_tc"] and c["record_in_tc"] for c in cuts))

    no_loop = build_cut_plan(RHYTHM_BEATS, video_duration=6.0,
                             audio_duration=10.0, ranks=[1], min_shot=0.8,
                             max_shot=3.0, loop=False, fps=24)
    check("sans boucle : timeline <= vidéo",
          no_loop["timeline_duration"] <= 6.0 + 1e-6,
          str(no_loop["timeline_duration"]))
    check("sans boucle : marque la troncature", no_loop["truncated"] is True)

    empty = build_cut_plan([], video_duration=6.0, audio_duration=10.0,
                           ranks=[1], min_shot=0.8, max_shot=3.0, loop=True)
    check("sans beat -> subdivise par la durée max", empty["segments"] == 4,
          str(empty["segments"]))
    check("sans beat -> aucun beat utilisé", empty["beats_used"] == 0)

    xml = export_rhythm("/tmp/ne_pas_un_fichier.mp4", cuts, 24, fmt="fcpxml",
                        audio_path="/tmp/musique.mp3", audio_duration=10.0)
    try:
        root = ET.fromstring(xml.split("\n", 2)[2])
        check("FCPXML rythme bien formé", True)
    except ET.ParseError as exc:
        check("FCPXML rythme bien formé", False, str(exc))
        return

    clips = root.findall(".//sequence/spine/asset-clip")
    check("un asset-clip par coupe + musique",
          len(clips) == len(cuts) + 1, str(len(clips)))
    check("musique sur une piste liée",
          any(c.get("lane") == "-1" for c in clips))
    check("deux assets (vidéo + musique)",
          len(root.findall(".//resources/asset")) == 2)
    check("durée de séquence renseignée",
          bool(root.find(".//sequence").get("duration")))

    sans_audio = export_rhythm("/tmp/ne_pas_un_fichier.mp4", cuts, 24,
                               fmt="fcpxml", include_audio=False)
    root2 = ET.fromstring(sans_audio.split("\n", 2)[2])
    check("export sans musique -> un seul asset",
          len(root2.findall(".//resources/asset")) == 1)

    edl = export_rhythm("/tmp/video.mp4", cuts, 24, fmt="edl")
    events = [l for l in edl.splitlines()
              if l[:3].isdigit() and "  AX" in l]
    check("EDL : un événement par coupe", len(events) == len(cuts),
          str(len(events)))
    check("EDL : 4 timecodes par événement",
          all(len(l.split()) == 8 for l in events))

    csv = export_rhythm("/tmp/video.mp4", cuts, 24, fmt="csv")
    check("CSV : en-tête + une ligne par coupe",
          len(csv.splitlines()) == len(cuts) + 1)

    try:
        export_rhythm("/tmp/video.mp4", cuts, 24, fmt="inconnu")
        check("format inconnu -> erreur", False)
    except ValueError:
        check("format inconnu -> erreur", True)


def test_api() -> None:
    print("\n[6] API HTTP (serveur local)")
    base = "http://127.0.0.1:8005"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as resp:
            return resp.status, resp.read()

    try:
        status, _ = get("/api/ping")
        check("ping", status == 200)
    except Exception as exc:
        check("ping", False, f"serveur injoignable ({exc})")
        return

    _, body = get("/api/config")
    cfg = json.loads(body)
    check("config expose version", bool(cfg.get("version")), cfg.get("version"))
    check("config expose workspace", bool(cfg.get("workspace")))

    _, body = get("/static/index.html")
    html = body.decode("utf-8")
    check("bouton Envoyer à DaVinci", 'onclick="installDavinci()"' in html)
    check("bouton Télécharger le .py",
          "onclick=\"exportBeats('davinci_script')\"" in html)
    check("rang 3 décoché", 'id="rank3">' in html)
    check("onglet Montage au rythme", 'data-tab="rhythm"' in html)
    check("bouton Générer le montage", 'onclick="startRhythm()"' in html)
    check("export FCPXML rythme", "exportRhythm('fcpxml')" in html)

    test_api_rhythm(base)


def _post_json(base: str, path: str, payload: dict, timeout: float = 30):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def test_api_rhythm(base: str) -> None:
    """Vérifie /api/rhythm/* avec un vrai couple vidéo + musique (ffmpeg)."""
    import shutil
    import subprocess
    import time

    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as resp:
            return resp.status, resp.read()

    if not shutil.which("ffmpeg"):
        check("rythme : ffmpeg disponible", False, "tests API ignorés")
        return

    with tempfile.TemporaryDirectory() as tmp:
        video = os.path.join(tmp, "plan.mp4")
        audio = os.path.join(tmp, "musique.wav")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24",
                 "-t", "3", "-pix_fmt", "yuv420p", video],
                check=True, timeout=60,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
                 audio],
                check=True, timeout=60,
            )
        except Exception as exc:
            check("rythme : médias de test générés", False, str(exc)[:80])
            return

        # Scan valide
        try:
            status, body = _post_json(base, "/api/rhythm/scan",
                                      {"video_path": video, "audio_path": audio})
            info = json.loads(body)
            check("rythme scan", status == 200)
            check("rythme scan : durées",
                  info["video"]["duration"] > 0 and info["audio"]["duration"] > 0)
        except Exception as exc:
            check("rythme scan", False, str(exc)[:80])
            return

        # Erreurs de validation
        try:
            _post_json(base, "/api/rhythm/scan",
                       {"video_path": video, "audio_path": "/tmp/absent.wav"})
            check("rythme scan 404 fichier absent", False)
        except urllib.error.HTTPError as exc:
            check("rythme scan 404 fichier absent", exc.code == 404,
                  str(exc.code))

        try:
            _post_json(base, "/api/rhythm/scan",
                       {"video_path": audio, "audio_path": video})
            check("rythme scan 400 mauvaise source", False)
        except urllib.error.HTTPError as exc:
            check("rythme scan 400 mauvaise source", exc.code == 400,
                  str(exc.code))

        # Job complet
        try:
            status, body = _post_json(base, "/api/rhythm/start", {
                "video_path": video, "audio_path": audio,
                "ranks": [1, 2], "min_shot": 0.4, "max_shot": 2.0,
                "loop": True, "fps": 24,
            })
            job = json.loads(body)["id"]
            check("rythme start", status == 200 and isinstance(job, int))
        except Exception as exc:
            check("rythme start", False, str(exc)[:80])
            return

        result = None
        for _ in range(120):
            time.sleep(0.25)
            _, body = get(f"/api/job/{job}")
            state = json.loads(body)
            if state.get("status") in ("done", "error", "cancelled"):
                result = state
                break
        check("rythme job terminé", bool(result) and result["status"] == "done",
              (result or {}).get("error") or (result or {}).get("status", "timeout"))
        if not result or result["status"] != "done":
            return

        res = result["result"]
        check("rythme : au moins un plan", res["segments"] >= 1,
              str(res["segments"]))
        check("rythme : timeline = musique",
              abs(res["timeline_duration"] - res["audio_duration"]) < 0.05,
              str(res["timeline_duration"]))

        for fmt, marker in (("fcpxml", "<fcpxml"), ("edl", "TITLE:"),
                            ("csv", "shot,beat_s")):
            try:
                status, body = _post_json(base, "/api/rhythm/export",
                                          {"job_id": job, "fmt": fmt})
                text = body.decode("utf-8")
                check(f"rythme export {fmt}",
                      status == 200 and marker in text, str(len(text)))
            except Exception as exc:
                check(f"rythme export {fmt}", False, str(exc)[:80])

        try:
            _post_json(base, "/api/rhythm/export", {"job_id": job, "fmt": "zip"})
            check("rythme export format invalide -> 400", False)
        except urllib.error.HTTPError as exc:
            check("rythme export format invalide -> 400", exc.code == 400,
                  str(exc.code))

        try:
            _post_json(base, "/api/rhythm/export", {"job_id": 999999,
                                                    "fmt": "csv"})
            check("rythme export job inconnu -> 404", False)
        except urllib.error.HTTPError as exc:
            check("rythme export job inconnu -> 404", exc.code == 404,
                  str(exc.code))


def test_alignment() -> None:
    """Détection audio : alignement à < 10 ms sur une piste de clics synthétiques."""
    print("\n[6] Alignement audio")
    import tempfile
    from pathlib import Path
    try:
        import numpy as np
        import soundfile as sf
        from app.services.audio_beats import analyze_audio
    except Exception as exc:
        check("dépendances disponibles", False, str(exc))
        # pas de suites
        return

    sr = 22050
    dur = 10.0
    y = np.zeros(int(sr * dur), dtype=np.float32)
    rng = np.random.default_rng(42)
    truth = [0.37, 1.12, 2.53, 3.78, 5.01, 6.24, 7.5, 8.77]
    for t in truth:
        idx = int(round(t * sr))
        n = int(0.02 * sr)
        env = np.exp(-np.arange(n) / (0.008 * sr))
        tone = (np.sin(2 * np.pi * 220 * np.arange(n) / sr)
                + 0.6 * (rng.random(n) * 2 - 1)) * env
        y[idx:idx + n] += tone.astype(np.float32) * 0.8

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "clicks.wav")
        sf.write(path, y, sr)
        res = analyze_audio(path, sr=sr)
        det = sorted(b["time"] for b in res["beats"])
        errs = [(min(det, key=lambda d: abs(d - t)) - t) * 1000 for t in truth]
        worst = max(abs(e) for e in errs) if errs else 0.0
        med = float(np.median(errs))
        check("détection : variations détectées",
              len(det) >= len(truth) and len(det) <= len(truth) + 3,
              f"{len(det)} détectés / {len(truth)} attendus")
        check("médiane erreur < 10 ms", abs(med) < 10, f"{med:+.1f} ms")
        check("aucun marqueur décalé de > 15 ms", worst < 15.0, f"{worst:.1f} ms")

    from app.utils import seconds_to_edl_timecode, seconds_to_frames
    fps = 24
    for s in (0.0, 0.116, 0.5, 1.0):
        f = seconds_to_frames(s, fps)
        tc = seconds_to_edl_timecode(s, fps)
        expected = "{:02d}:{:02d}:{:02d}:{:02d}".format(
            f // (fps * 3600),
            (f // (fps * 60)) % 60,
            (f // fps) % 60, f % fps)
        check("timecode {:.3f}s -> {}".format(s, tc), tc == expected)


def main() -> int:
    print("EdittingHub — tests de fumée")
    test_script()
    test_fcpxml()
    test_fcpxml_scene_valid()
    test_exports()
    test_install()
    test_alignment()
    test_rhythm()
    if "--api" in sys.argv:
        test_api()

    total = len(_results)
    echecs = [label for ok, label in _results if not ok]
    print(f"\n{total - len(echecs)}/{total} tests OK")
    if echecs:
        print("Échecs :")
        for label in echecs:
            print("  -", label)
        return 1
    print("Tout est vert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
