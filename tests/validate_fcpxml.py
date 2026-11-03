#!/usr/bin/env python3
"""Validateur FCPXML 1.9 pour l'import DaVinci Resolve.

Usage :
    python tests/validate_fcpxml.py [fichier|dossier ...]

Sans argument, scanne `outputs/` (et les sous-dossiers) du projet.

Vérifie, pour chaque fichier .fcpxml :
  - XML bien formé, déclaration UTF-8, version="1.9"
  - racine fcpxml, resources > format (frameDuration rationnel),
    asset (duration rationnel, hasVideo/hasAudio), media-rep (kind + src file://)
  - library/event/project/sequence et ses attributs requis
  - asset-clip de la spine : offset/start/duration rationnels, in-bounds,
    pas d'offset dupliqué, pas de durée < 1 frame
  - durée de séquence = somme des durées des clips
  - longueur totale = somme des durées des clips
  - noms sans saut de ligne
  - chemins src existants après unquote
"""
from __future__ import annotations

import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent

RATIONAL_RE = re.compile(r"^(\d+(?:/\d+)?)s$")


def _parse_rational(value):
    """Parse '12/24s' ou '0s' → Fraction. Lève ValueError si invalide."""
    if value is None:
        raise ValueError("valeur None")
    text = str(value).strip()
    m = RATIONAL_RE.match(text)
    if not m:
        raise ValueError(f"rationnel invalide : {value!r}")
    core = text[:-1]  # sans 's'
    if "/" in core:
        a, b = core.split("/", 1)
        num, den = int(a), int(b)
    else:
        num, den = int(core), 1
    if den <= 0:
        raise ValueError(f"dénominateur non positif : {value!r}")
    return Fraction(num, den)


def _frames(seconds: Fraction, frame: Fraction) -> int:
    """Nombre de frames arrondis pour une durée."""
    if frame <= 0:
        return 0
    return int((seconds / frame) + Fraction(1, 2))  # floor-ish via Fraction


def _collect_files(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for root, _, names in os.walk(p):
                for n in sorted(names):
                    if n.lower().endswith(".fcpxml"):
                        files.append(Path(root) / n)
        elif p.is_file():
            files.append(p)
    # déduplique en conservant l'ordre
    seen = set()
    out = []
    for f in files:
        rf = f.resolve()
        if rf in seen:
            continue
        seen.add(rf)
        out.append(f)
    return out


def validate_file(path) -> list[str]:
    errors: list[str] = []
    path = Path(path)

    def fail(msg):
        errors.append(f"{path.name}: {msg}")

    # Encodage / déclaration
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"{path}: impossible à lire ({exc})"]
    text_head = raw[:200].decode("ascii", errors="replace")
    if "encoding=\"UTF-8\"" not in text_head and "encoding='UTF-8'" not in text_head:
        fail("déclaration UTF-8 manquante ou incorrecte")
    if "version=\"1.0\"" not in text_head:
        fail("déclaration XML 1.0 manquante")

    # Parsing
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        fail(f"XML mal formé : {exc}")
        return errors

    if root.tag != "fcpxml":
        fail(f"racine inattendue : {root.tag}")
        return errors
    if root.get("version") != "1.9":
        fail(f"version != 1.9 ({root.get('version')!r})")

    # resources
    resources = root.find("resources")
    if resources is None:
        fail("élément <resources> manquant")
        return errors

    fmt_el = resources.find("format")
    if fmt_el is None:
        fail("<format> manquant")
        frame_duration = None
    else:
        fd_str = fmt_el.get("frameDuration")
        try:
            frame_duration = _parse_rational(fd_str)
        except ValueError as exc:
            fail(f"frameDuration invalide : {exc}")
            frame_duration = None
        if frame_duration is not None and frame_duration <= 0:
            fail("frameDuration <= 0")

    assets = resources.findall("asset")
    if not assets:
        fail("<asset> manquant")

    # Vérifie l'uniquité des id? non demandé. On vérifie src + bounds via asset_duration.
    asset_durations = {}
    for asset in assets:
        aid = asset.get("id") or "r2"
        try:
            adur = _parse_rational(asset.get("duration"))
        except ValueError as exc:
            fail(f"asset {aid} duration invalide : {exc}")
            adur = None
        if adur is not None and adur <= 0:
            fail(f"asset {aid} duration <= 0")
        # hasVideo/hasAudio
        hv = asset.get("hasVideo")
        ha = asset.get("hasAudio")
        if hv not in ("0", "1"):
            fail(f"asset {aid} hasVideo invalide ({hv!r})")
        if ha not in ("0", "1"):
            fail(f"asset {aid} hasAudio invalide ({ha!r})")
        if asset.get("name") and "\n" in asset.get("name"):
            fail(f"asset {aid} name contient un saut de ligne")
        # media-rep
        reps = asset.findall("media-rep")
        if not reps:
            fail(f"asset {aid} sans media-rep")
        for rep in reps:
            if rep.get("kind") != "original-media":
                fail(f"asset {aid} media-rep kind != original-media ({rep.get('kind')!r})")
            src = rep.get("src")
            if not src or not src.startswith("file://"):
                fail(f"asset {aid} src n'est pas file:// ({src!r})")
            else:
                fpath = unquote(src[len("file://"):])
                # file:// + chemin absolu → enlever les '//' supplémentaires
                if fpath.startswith("//"):
                    fpath = fpath[1:]
                if not os.path.exists(fpath):
                    fail(f"asset {aid} src inexistant : {fpath}")
        if adur is not None:
            asset_durations[aid] = adur
        # format requis si hasVideo
        if hv == "1" and not asset.get("format"):
            fail(f"asset {aid} (hasVideo=1) sans attribut format")

    # library/event/project/sequence
    library = root.find("library")
    if library is None:
        fail("<library> manquant")
    event = library.find("event") if library is not None else None
    if event is None:
        fail("<event> manquant")
    project = event.find("project") if event is not None else None
    if project is None:
        fail("<project> manquant")
    sequence = project.find("sequence") if project is not None else None
    if sequence is None:
        fail("<sequence> manquant")
        return errors

    # attrs sequence requis
    for attr in ("format", "duration", "tcStart", "tcFormat"):
        if sequence.get(attr) is None:
            fail(f"sequence attribut requis manquant : {attr}")
    try:
        seq_dur = _parse_rational(sequence.get("duration"))
    except ValueError as exc:
        fail(f"sequence duration invalide : {exc}")
        seq_dur = None

    # spine
    spine = sequence.find("spine")
    if spine is None:
        fail("<spine> manquant dans sequence")
        return errors

    clips = spine.findall("asset-clip")
    offsets = []
    clip_durations = []
    for idx, clip in enumerate(clips, 1):
        ref = clip.get("ref")
        name = clip.get("name")
        if name and "\n" in name:
            fail(f"clip {idx} name contient un saut de ligne")
        # rational attrs
        for attr in ("offset", "start", "duration"):
            val = clip.get(attr)
            try:
                _parse_rational(val)
            except (ValueError, TypeError) as exc:
                fail(f"clip {idx} {attr} invalide ({val!r}) : {exc}")
        try:
            off = _parse_rational(clip.get("offset"))
            st = _parse_rational(clip.get("start"))
            dur = _parse_rational(clip.get("duration"))
        except (ValueError, TypeError):
            continue
        offsets.append(off)
        clip_durations.append(dur)

        # frames
        if frame_duration is not None and frame_duration > 0:
            frames = dur / frame_duration
            if frames < 1:
                fail(f"clip {idx} durée < 1 frame ({frames})")
            try:
                start_frames = int(st / frame_duration) if False else None
            except Exception:
                start_frames = None

        # in-bounds vs asset duration
        adur = asset_durations.get(ref)
        if adur is not None:
            if not (Fraction(0) <= st <= adur + (frame_duration or Fraction(0))):
                fail(f"clip {idx} start hors bornes (start={st}, asset={adur})")
            end = st + dur
            if end > adur + (frame_duration or Fraction(1, 24)):
                fail(f"clip {idx} dépasse l'asset (end={end}, asset={adur})")

        # Rational components match : chaque durée tombe sur un nombre entier
        # de frames selon frameDuration (dénominateur aligné).
        if frame_duration is not None and frame_duration > 0:
            for attr in ("offset", "start", "duration"):
                try:
                    r = _parse_rational(clip.get(attr))
                except ValueError:
                    continue
                if r == 0:
                    continue
                frac_frames = r / frame_duration
                if frac_frames.denominator != 1:
                    fail(f"clip {idx} {attr} ne tombe pas sur un frame entier "
                         f"({r} / {frame_duration} = {frac_frames})")
                # Le dénominateur brut (non réduit) peut différer après
                # simplification Fraction : on vérifie déjà l'intégralité en
                # frames ci-dessus, ce qui est le critère déterminant.
        # markers (optionnels) : pas de newline dans value
        for marker in clip.findall("marker"):
            val = marker.get("value")
            if val and "\n" in val:
                fail(f"clip {idx} marker value avec saut de ligne")
            # duration marker = 1 frame
            try:
                md = _parse_rational(marker.get("duration"))
                if frame_duration is not None and md != frame_duration:
                    # tolérant : on n'échoue pas, juste info
                    pass
            except ValueError:
                fail(f"clip {idx} marker duration invalide")

    # no duplicate offset
    seen_offsets = set()
    for off in offsets:
        if off in seen_offsets:
            fail(f"offset dupliqué : {off}")
        seen_offsets.add(off)

    # sequence duration = sum durations (rationnel exact)
    total_clip_duration = sum(clip_durations, Fraction(0))
    if seq_dur is not None and clip_durations:
        if seq_dur != total_clip_duration:
            fail(f"sequence duration ({seq_dur}) != somme des clips ({total_clip_duration})")
    # total length seconds == sum clip durations
    # (déjà couvert par seq_dur ; on revérifie en secondes)
    if seq_dur is not None and clip_durations:
        if abs(float(seq_dur) - float(total_clip_duration)) > 1e-9:
            fail("longueur totale != somme des durées des clips (secondes)")

    # markers anywhere no newline in name/value
    for el in root.iter():
        nm = el.get("name")
        if nm and "\n" in nm:
            fail(f"élément {el.tag} name avec saut de ligne")
        vl = el.get("value")
        if vl and "\n" in vl:
            fail(f"élément {el.tag} value avec saut de ligne")

    return errors


def main(argv) -> int:
    if len(argv) > 1:
        targets = argv[1:]
    else:
        targets = [str(ROOT / "outputs")]

    files = _collect_files(targets)
    if not files:
        print("Aucun fichier .fcpxml trouvé.")
        return 0

    total_errors = 0
    for f in files:
        errs = validate_file(f)
        if errs:
            total_errors += len(errs)
            print(f"FAIL {f}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {f}")

    print(f"\n{len(files)} fichier(s) vérifié(s), {total_errors} erreur(s).")
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
