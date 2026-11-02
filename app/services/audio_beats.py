"""Analyseur et Classificateur de Beats Audio — utilise Librosa en local.

Analyse la piste audio d'une musique pour :
  1. Extraire les onsets (transitoires / percussions) ;
  2. Classifier les beats en 3 niveaux stricts :
     - Rang 1 : Beats principaux / impacts forts (amplitude maximale) ;
     - Rang 2 : Beats secondaires / intermédiaires ;
     - Rang 3 : Contre-temps / micro-variations ;
  3. Exporter les marqueurs audio sous forme de CSV ou EDL compatible
     avec les marqueurs de timeline DaVinci Resolve.

100 % local, algorithmique (pas d'inférence LLM).
Matériel : Ryzen 5 5600G + RX 6600 — librosa tourne sur CPU (rapide
pour l'analyse d'onset, < 5 s pour un morceau de 4 min).
"""
import os
import time

from app.config import (DEFAULT_AUDIO_SR, DEFAULT_RANK1_PERCENTILE,
                        DEFAULT_RANK2_PERCENTILE, MIN_BEAT_INTERVAL,
                        RANK_COLORS)
from app.utils import seconds_to_edl_timecode


def _load_audio(audio_path, sr=DEFAULT_AUDIO_SR, cancel_event=None):
    """Charge un fichier audio/vidéo avec librosa.

    Tous les formats (mp3, wav, flac, m4a, mp4, mkv, …) sont d'abord
    convertis en WAV PCM 16kHz mono via ffmpeg, puis chargés par librosa.
    Cette approche garantit la compatibilité (soundfile ne lit pas MP3/MP4)
    et évite les erreurs de décodage.

    L'extraction ffmpeg est tuée immédiatement si `cancel_event` est levé.
    """
    import librosa
    import tempfile
    import subprocess
    import time as _time

    # Étape 1 : extraire/convertir l'audio en WAV temporaire
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", audio_path, "-ar", str(sr), "-ac", "1",
               "-c:a", "pcm_s16le", wav_path]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg introuvable — installez-le (sudo apt install ffmpeg)."
            )

        started = _time.time()
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise JobCancelled("Extraction audio annulée.")
            if _time.time() - started > 120:
                proc.kill()
                raise RuntimeError("Timeout de l'extraction audio (120s).")
            _time.sleep(0.1)

        _out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg ne peut extraire l'audio de {audio_path} : "
                f"{(err or b'').decode(errors='replace')[:200]}"
            )
    except Exception:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise

    # Étape 2 : charger le WAV avec librosa (soundfile supporte bien le PCM)
    try:
        y, sr_actual = librosa.load(wav_path, sr=sr, mono=True, duration=None)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    return y, sr_actual


def _compute_onsets(y, sr):
    """Calcule l'enveloppe d'onset et détecte les onsets.

    Retourne : onset_envelope, onset_times, onset_strengths
    """
    import numpy as np
    import librosa

    # Enveloppe d'onset (mel-spectrogram + flux temporel)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate="mean")

    # Temps des onsets (transitoires)
    # Les paramètres de peak_pick sont en *frames* (hop_length=512 par défaut)
    hop = 512
    min_interval_frames = max(1, int(MIN_BEAT_INTERVAL * sr / hop))
    onset_times = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units="time",
        pre_max=min_interval_frames,
        post_max=min_interval_frames,
        delta=0.05,          # seuil relatif (plus sensible)
        wait=1,
    )

    if len(onset_times) == 0:
        return onset_env, np.array([]), np.array([])

    # Amplitude/peak d'énergie à chaque onset
    # onset_env est indexé par frames de 512 échantillons
    hop = 512
    onset_samples = (onset_times * sr).astype(int)
    onset_frames = onset_samples // hop
    onset_frames = np.clip(onset_frames, 0, len(onset_env) - 1)
    onset_strengths = onset_env[onset_frames]

    return onset_env, onset_times, onset_strengths


def _estimate_tempo_sr(onset_env, sr):
    """Estime le tempo et le rayon de beat tracking.

    Utilise beat_track (tempo + positions de beats) puis PLP pour
    une activation de beat plus robuste.

    Retourne (tempo: float, plp: np.ndarray).
    """
    import numpy as np
    import librosa

    # Tempo estimé via beat_track
    try:
        tempo_values, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, units="frames"
        )
        # tempo_values peut être un array (si plusieurs tempos) ou un scalaire
        tempo = float(np.mean(tempo_values)) if len(tempo_values) else 0.0
        if tempo <= 0:
            tempo = 120.0
    except Exception:
        tempo = 120.0

    # PLP beat activation (pour une grille de beats plus fine)
    try:
        plp = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    except TypeError:
        # Signature différente selon les versions
        plp = onset_env

    return float(tempo), plp


def _classify_beats(onset_times, onset_strengths, tempo, sr):
    """Classifie les onsets en 3 rangs selon l'énergie.

    Rang 1 : énergie > percentile 66  → impacts forts / beats principaux
    Rang 2 : énergie > percentile 33  → beats secondaires / intermédiaires
    Rang 3 : énergie <= percentile 33 → contre-temps / micro-variations

    Bonus : les onsets qui tombent sur une grille de beat (multiples de
    60/tempo secondes) sont favorisés vers le rang 1.
    """
    import numpy as np

    beats = []
    if len(onset_times) == 0:
        return beats

    strengths = np.array(onset_strengths, dtype=np.float64)
    times = np.array(onset_times, dtype=np.float64)

    # Percentiles dynamiques
    p1 = np.percentile(strengths, DEFAULT_RANK1_PERCENTILE) if len(strengths) > 0 else 1.0
    p2 = np.percentile(strengths, DEFAULT_RANK2_PERCENTILE) if len(strengths) > 0 else 0.5

    # Grille de beats principale (period en secondes)
    period = 60.0 / tempo if tempo > 0 else 0.5

    for t, s in zip(times, strengths):
        # Bonus : si l'onset tombe sur la grille de beat, élève le rang
        if period > 0:
            beat_idx = round(t / period)
            nearest_beat = beat_idx * period
            if abs(t - nearest_beat) < period * 0.15:  # ±15% de la période
                # Sur la grille → favorise rang 1 ou 2
                if s >= p2:
                    rank = 1
                else:
                    rank = 2
            else:
                # Hors grille → classification par énergie
                if s >= p1:
                    rank = 1
                elif s >= p2:
                    rank = 2
                else:
                    rank = 3
        else:
            # Classification par énergie uniquement
            if s >= p1:
                rank = 1
            elif s >= p2:
                rank = 2
            else:
                rank = 3

        beats.append({
            "time": round(float(t), 3),
            "energy": round(float(s), 4),
            "rank": rank,
            "color": RANK_COLORS.get(rank, "White"),
            "label": {1: "Impact fort", 2: "Intermédiaire", 3: "Micro-variation"}[rank],
        })

    return beats


def analyze_audio(audio_path, sr=DEFAULT_AUDIO_SR, progress_cb=None,
                  cancel_event=None):
    """Analyse complète de la piste audio.

    Args:
        audio_path : chemin vers le fichier audio/vidéo
        sr : sample rate de travail (22050 par défaut)
        progress_cb : callback(frac, step) — progression par étapes
        cancel_event : threading.Event pour annuler (kill de ffmpeg inclus)

    Returns:
        dict avec : beats, fps, duration, tempo, total_onsets, elapsed
    """
    from app.errors import JobCancelled

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {audio_path}")
    if cancel_event is not None and cancel_event.is_set():
        raise JobCancelled("Analyse annulée avant démarrage.")

    def step(frac, label):
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled("Analyse audio annulée.")
        if progress_cb:
            progress_cb(frac, label)

    t0 = time.time()

    # Chargement (extraction ffmpeg : ~10 % du travail)
    step(0.02, "Extraction de l'audio (ffmpeg)")
    y, sr_actual = _load_audio(audio_path, sr, cancel_event=cancel_event)
    duration = librosa_seconds(y, sr_actual)
    step(0.25, f"Audio chargé ({duration:.1f} s, {sr_actual} Hz)")

    # Enveloppe d'onset + détection
    step(0.3, "Détection des onsets")
    onset_env, onset_times, onset_strengths = _compute_onsets(y, sr_actual)
    step(0.7, f"{len(onset_times)} onsets détectés")

    # Tempo estimé
    step(0.75, "Estimation du tempo")
    tempo, _plp = _estimate_tempo_sr(onset_env, sr_actual)

    # Classification
    step(0.85, "Classification des beats (3 rangs)")
    beats = _classify_beats(onset_times, onset_strengths, tempo, sr_actual)
    step(1.0, f"{len(beats)} beats classifiés")

    elapsed = time.time() - t0

    return {
        "status": "done",
        "beats": beats,
        "tempo": round(tempo, 1),
        "duration": round(duration, 3),
        "total_onsets": len(onset_times),
        "total_beats": len(beats),
        "ranks": {
            1: sum(1 for b in beats if b["rank"] == 1),
            2: sum(1 for b in beats if b["rank"] == 2),
            3: sum(1 for b in beats if b["rank"] == 3),
        },
        "sr": sr_actual,
        "elapsed": round(elapsed, 2),
        "audio": os.path.basename(audio_path),
    }


def librosa_seconds(y, sr):
    """Calcule la durée en secondes d'un tableau audio."""
    return len(y) / sr if sr > 0 else 0.0


# ── Export CSV ─────────────────────────────────────────────────────────────

def export_beats_csv(beats, fps=24.0):
    """Export CSV pour DaVinci Resolve markers (format séparateur point-virgule).

    Colonnes : Name, Start (s), End (s), Duration, Color, Comment
    Les temps sont en secondes (Resolve accepte les CSV en secondes).
    """
    lines = ["Name;Start (s);End (s);Duration;Color;Comment"]
    for i, b in enumerate(beats, 1):
        start = b["time"]
        end = start + 0.1  # durée arbitraire (100 ms)
        name = f"Beat {i} — {b['label']}"
        comment = f"Rank {b['rank']} | Energy {b['energy']} | {start:.3f}s"
        lines.append(
            f"{name};{start:.3f};{end:.3f};0.10;{b['color']};{comment}"
        )
    return "\n".join(lines)


# ── Export EDL avec markers ────────────────────────────────────────────────

def export_beats_edl(beats, fps=24.0, title=None):
    """Export EDL CMX 3600 avec events de type M (markers).

    Chaque beat devient un marker M avec un COMMENT.
    DaVinci Resolve peut importer cet EDL et placer des marqueurs à chaque timecode.
    """
    title = title or "Audio Beat Markers"
    lines = []
    lines.append(f"TITLE: {title}")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    for i, b in enumerate(beats, 1):
        tc = seconds_to_edl_timecode(b["time"], fps)
        # Event M (marker) — un seul frame de durée
        end_tc = seconds_to_edl_timecode(b["time"] + 1.0 / fps, fps)
        lines.append(f"{i:03d}  M      V     C        {tc} {end_tc}")
        lines.append(
            f"     * COMMENT: Level {b['rank']} - {b['label']} "
            f"({b['time']:.3f}s, energy={b['energy']})"
        )
        lines.append("")

    return "\n".join(lines)


# ── Export XML (FCPXML) avec markers ────────────────────────────────────────

def export_beats_xml(beats, fps=24.0, title=None, asset_path=None):
    """Export FCPXML 1.9 avec markers posés sur le clip source.

    L'ancienne version plaçait les `<marker>` directement sous `<sequence>`
    (invalide) et déclarait un `<asset>` sans `src` ni `duration` : Resolve
    importait alors une timeline vide. On délègue à `app.fcpxml` qui produit
    une structure valide (asset + media-rep + spine + asset-clip + markers).

    Args:
        beats : liste de dicts {time, label, rank, energy, …}
        fps : fps de la timeline
        title : nom du projet dans Resolve
        asset_path : média réel à référencer (fortement recommandé : sans lui,
                     le clip ne pointe vers aucun fichier et Resolve n'affiche
                     que des marqueurs sans média).
    """
    from app.fcpxml import markers_document

    markers = [
        {
            "time": float(b.get("time") or 0.0),
            "value": f"Beat {i} — {b.get('label', '')}".strip(),
            "note": (f"Rang {b.get('rank')} | énergie {b.get('energy')} "
                     f"| {float(b.get('time') or 0.0):.3f}s"),
        }
        for i, b in enumerate(beats, 1)
    ]
    total = (float(beats[-1]["time"]) + 1.0) if beats else 1.0
    return markers_document(asset_path or "", fps, markers, title=title,
                            duration=total)


def _resolve_marker_color(rank):
    """Mappe le rang du beat à la couleur Resolve (info interne / CSV)."""
    return {1: "Red", 2: "Yellow", 3: "Blue"}.get(rank, "White")


# ── Export DaVinci Resolve Python Script ────────────────────────────────────

def export_beats_davinci_script(beats, fps=24.0, title=None, asset_path=None, ranks=None):
    """Génère le script Python autonome pour DaVinci Resolve.

    Le script est exécutable depuis *Workspace → Scripts* (installé par
    ``/api/audio/davinci/install``) ou collé dans *Workspace → Console →
    onglet Python*. Voir ``app/davinci.py`` pour la stratégie de robustesse.

    Args:
        beats : liste de dicts {time, label, rank, energy, …} — DÉJÀ FILTRÉS
        fps : fps de la timeline (repli si Resolve n'en fournit pas)
        title : nom du projet (informatif)
        asset_path : média source (informatif)
        ranks : rangs exportés (informatif)
    """
    from app.davinci import build_script

    return build_script(beats, fps=fps, title=title, asset_path=asset_path,
                        ranks=ranks)


# ── Fonction unifiée d'export ────────────────────────────────────────────────

def export_markers(beats, fps=24.0, fmt="csv", title=None, asset_path=None, ranks=None):
    """Exporte les beats dans le format demandé.

    fmt : 'csv' | 'edl' | 'xml' | 'fcpxml' | 'davinci_script'
    asset_path : média référencé par le XML (obligatoire pour un import
                 exploitable dans Resolve)
    ranks : liste des rangs à inclure (ex: [1,2]) — None = tous
    Retourne le contenu texte.
    """
    if ranks is not None:
        beats = [b for b in beats if int(b.get("rank") or 0) in ranks]
    if fmt == "csv":
        return export_beats_csv(beats, fps)
    elif fmt == "edl":
        return export_beats_edl(beats, fps, title)
    elif fmt in ("xml", "fcpxml"):
        # xml et fcpxml génèrent tous deux du FCPXML 1.9 valide pour Resolve
        return export_beats_xml(beats, fps, title, asset_path=asset_path)
    elif fmt == "davinci_script":
        return export_beats_davinci_script(beats, fps, title, asset_path, ranks)
    else:
        raise ValueError(f"Format d'export inconnu : {fmt}")
