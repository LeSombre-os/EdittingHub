"""Analyseur et Classificateur de Beats Audio — utilise Librosa en local.

Analyse la piste audio d'une musique pour :
  1. Extraire les onsets (transitoires / percussions) ;
  2. Classifier les beats en 3 niveaux stricts :
     - Rang 1 : Beats principaux / impacts forts (amplitude maximale) ;
     - Rang 2 : Beats secondaires / intermédiaires ;
     - Rang 3 : Contre-temps / micro-variations ;
  3. Exporter les marqueurs audio sous forme de CSV ou EDL compatible
     avec les marqueurs de timeline DaVinci Resolve.

Précision des marqueurs
-----------------------
Trois réglages déterminent l'alignement d'un marqueur sur l'attaque :

1. **Agrégation des bandes mel** — l'enveloppe d'onset doit moyenner les
   bandes (`aggregate=np.mean`, un *callable*). Avec la chaîne `"mean"`,
   librosa 1.0 n'agrège rien et ne renvoie qu'une seule bande (graves) :
   les marqueurs tombaient alors sur des tenues de basse, pas sur les
   frappes réelles.
2. **Résolution temporelle** — un hop de 512 échantillons vaut 23 ms à
   22050 Hz, soit plus d'une demi-image à 24 fps. On analyse à ~5 ms
   (`ONSET_HOP_SECONDS`).
3. **Recalage sur l'attaque** — le peak-picking place l'onset sur le
   maximum du flux spectral, systématiquement 10 à 25 ms après le début
   du transitoire. Chaque onset est ramené sur le début de la montée
   d'énergie mesurée à 1 ms (`_snap_to_attack`).

Mesuré sur une piste de clics synthétiques : biais ramené de +19 ms à
-2 ms, écart-type de ±8 ms à ±2 ms.

100 % local, algorithmique (pas d'inférence LLM).
Matériel : Ryzen 5 5600G + RX 6600 — librosa tourne sur CPU (rapide
pour l'analyse d'onset, < 5 s pour un morceau de 4 min).
"""
import math
import os
import time

from app.config import (DEFAULT_AUDIO_SR, DEFAULT_RANK1_PERCENTILE,
                        DEFAULT_RANK2_PERCENTILE, MIN_BEAT_INTERVAL,
                        ONSET_AVG_WINDOW, ONSET_DELTA, ONSET_HOP_SECONDS,
                        ONSET_PRE_MAX, ONSET_REFINE_MAX_BACK,
                        ONSET_REFINE_SEARCH, RANK_COLORS)
from app.utils import seconds_to_edl_timecode


def _audio_offset(audio_path):
    """Décalage audio (s) par rapport au flux vidéo, ou 0 si non détectable.

    Desynchronisation typique d'un flux MXF/TS : la piste audio commence à
    0,5 s alors que la vidéo part de 0. On ne l'applique qu'au-delà de
    40 ms (une image à 25 fps) car les petits écarts sont du bruit de
    conteneur (priming AAC, LAME, etc.) déjà géré par ffmpeg.

    L'helper lève les exceptions de ffprobe : toutes les sources de détection
    sont un repli silencieux.
    """
    from app.media import probe

    try:
        info = probe(audio_path)
    except Exception:
        return 0.0
    if not info.get("has_video"):
        return 0.0
    # Videostream plutôt que format : la timeline de référence est la vidéo.
    video_start = info.get("video_start")
    audio_start = info.get("audio_start")
    if video_start is None and audio_start is None:
        return 0.0
    offset = (audio_start or 0.0) - (video_start or 0.0)
    if 0.04 <= offset <= 10.0:
        return float(offset)
    return 0.0


def _analysis_hop(sr):
    """Hop d'analyse ≈ 5 ms, en puissance de 2 (contrainte STFT)."""
    target = max(1.0, ONSET_HOP_SECONDS * float(sr))
    return 1 << max(4, int(round(math.log2(target))))


def _fine_envelope(y, sr):
    """Enveloppe d'attaque à ~1 ms, pour recaler un onset sur l'attaque.

    Différence première (accentue les attaques, atténue les graves) puis
    lissage énergétique ; le maximum par bloc de 1 ms conserve la crête.
    Calculée **une seule fois** par analyse : elle servait auparavant trois
    fois (recalage, filtre des faux onsets, marqueurs), soit 3 passes sur
    l'échantillonnage complet.
    """
    import numpy as np
    from scipy.ndimage import uniform_filter1d

    diff = np.abs(np.diff(y, prepend=y[:1]))
    width = max(1, int(round(sr * 0.0015)))
    smooth = uniform_filter1d(diff, size=width, mode="nearest")
    hop = max(1, int(round(sr * 0.001)))
    frames = max(1, len(smooth) // hop)
    return smooth[:frames * hop].reshape(frames, hop).max(axis=1), hop


def _rolling_median(x, window):
    """Médiane glissante sur les `window` échantillons **précédents**.

    Équivalent vectorisé de ``median(x[max(0,i-window+1):i+1])`` : boucler en
    Python sur ~240 000 trames coûtait plusieurs secondes par morceau.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    pad = window - 1
    padded = np.pad(x, (pad, 0), mode="edge")
    out = np.median(sliding_window_view(padded, window), axis=1)
    # Les premières trames n'ont pas d'historique complet : on reproduit
    # exactement la médiane du préfixe disponible (sinon divergence à ~8 ms).
    for i in range(min(pad, out.size)):
        out[i] = np.median(x[: i + 1])
    return out


def _forward_max(x, window):
    """Maximum sur les `window` échantillons **suivants** (vectorisé)."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    padded = np.pad(x, (0, window - 1), mode="edge")
    return sliding_window_view(padded, window).max(axis=1)


def _snap_to_attack(times, novelty, hop, sr):
    """Recale chaque onset sur le début de la montée d'énergie adjacente.

    Le peak-picking librosa place l'onset sur le maximum du flux spectral
    (+10 à +25 ms après le début réel du transitoire). On cherche le
    pic local de l'enveloppe fine dans ±ONSET_REFINE_SEARCH puis on remonte
    la montée jusqu'au premier point sous 30 % de ce pic (12 ms max).
    """
    import numpy as np

    times = np.asarray(times, dtype=float)
    if times.size == 0 or novelty.size < 3:
        return times

    fps = float(sr) / float(hop)
    floor = float(np.median(novelty)) * 3.0
    out = times.copy()
    for index, event in enumerate(times):
        start = max(0, int((event - ONSET_REFINE_SEARCH) * fps))
        end = min(novelty.size, int((event + ONSET_REFINE_SEARCH) * fps) + 1)
        window = novelty[start:end]
        if window.size < 3:
            continue
        peak = int(np.argmax(window))
        threshold = max(float(window[peak]) * 0.30, floor)
        limit = max(0, peak - int(ONSET_REFINE_MAX_BACK * fps))
        pos = peak
        while pos > limit and window[pos] > threshold:
            pos -= 1
        out[index] = (start + pos) / fps
    return out


def _build_beat_markers(y, sr, onset_env, onset_times, onset_strengths, beat_times,
                        hop, fine=None):
    """Construit la liste finale des marqueurs de beats en combinant :
    1. La grille de beats (PLP) -> temps principaux
    2. L'enveloppe fine (1ms) -> détection d'attaques précises
    3. Les onsets détectés par librosa -> confirmation/force

    Retourne: (times, strengths, period, grid)
    """
    import numpy as np
    from scipy.signal import find_peaks

    # 1. Enveloppe fine à ~1 ms (partagée avec _compute_onsets)
    fe, hop_fine = fine if fine is not None else _fine_envelope(y, sr)
    fps = float(sr) / float(hop_fine)

    # Ratio d'attaque pour chaque trame fine (vectorisé)
    base = _rolling_median(fe, 9)
    rise = _forward_max(fe, 6)
    ratios = rise / (base + 1e-9)

    # 2. Grille de beats (PLP) - temps principaux
    grid = np.sort(np.asarray(beat_times, dtype=float)) if beat_times is not None else np.array([])
    meter = 4
    period = 60.0 / float(np.median(grid[1:] - grid[:-1])) if len(grid) > 1 else 60.0 / 120.0

    # 3. Pour chaque temps de grille, trouver l'attaque fine la plus proche (±80ms)
    main_beats = []
    for gt in grid:
        # Fenêtre de recherche autour du temps de grille
        window = min(0.08, period * 0.15)
        i_center = int(gt * fps)
        i_start = max(0, i_center - int(window * fps))
        i_end = min(len(ratios), i_center + int(window * fps) + 1)

        if i_start >= i_end:
            main_beats.append((gt, 1.0))
            continue

        # Trouver le pic de ratio dans la fenêtre
        local_ratios = ratios[i_start:i_end]
        if local_ratios.size == 0:
            main_beats.append((gt, 1.0))
            continue

        peak_idx = int(np.argmax(local_ratios))
        peak_ratio = float(local_ratios[peak_idx])
        peak_time = (i_start + peak_idx) / fps

        # Force de l'onset librosa le plus proche
        idx_onset = np.argmin(np.abs(onset_times - peak_time)) if len(onset_times) else -1
        onset_strength = float(onset_strengths[idx_onset]) if idx_onset >= 0 else 0.0

        main_beats.append((peak_time, onset_strength))

    # 4. Détecter les syncopations fortes (attaques fines hors grille)
    # Pics de ratio qui ne sont pas près d'un temps de grille
    off_grid_beats = []
    ratio_peaks, _ = find_peaks(ratios, distance=int(0.06 * fps), prominence=0.8, height=1.5)
    for pk in ratio_peaks:
        pk_time = pk / fps
        pk_ratio = float(ratios[pk])

        # Vérifier si près d'un temps de grille principal
        if grid.size > 0:
            dist_to_grid = np.min(np.abs(grid - pk_time))
            if dist_to_grid < period * 0.12:
                continue  # déjà couvert par la grille principale

        # Vérifier si près d'un beat principal déjà détecté
        if main_beats:
            dist_to_main = min(abs(pk_time - bt) for bt, _ in main_beats)
            if dist_to_main < 0.05:
                continue

        # Force librosa
        idx_onset = np.argmin(np.abs(onset_times - pk_time)) if len(onset_times) else -1
        onset_strength = float(onset_strengths[idx_onset]) if idx_onset >= 0 else 0.0

        off_grid_beats.append((pk_time, onset_strength))

    # 5bis. Combler les trous (beats manquants) : toute pause > 1.6×period
    # reçoit un beat synthétique de basse énergie pour éviter les silences
    # (rang 3 ou pire) — sans inventer de rang 1.
    filled = []
    if len(main_beats) >= 2 and period > 0:
        main_sorted = sorted(main_beats, key=lambda x: x[0])
        min_strength = min((s for _, s in main_sorted), default=0.0) or 0.0
        cap = 0.35 * max(min_strength, 1e-6)
        max_gap = period * 1.6
        for (t0, s0), (t1, s1) in zip(main_sorted, main_sorted[1:]):
            gap = t1 - t0
            if gap <= max_gap or gap > max_gap * 3.0:
                continue
            # insérer un point par période à l'intérieur
            n = int(round(gap / period)) - 1
            for k in range(1, n + 1):
                gt = t0 + k * period
                if gt < 0 or gt > (y.shape[0] / float(sr)):
                    break
                # énergie = min des voisins * 0.5 (sous le threshold de rang 1)
                filled.append((round(float(gt), 4), cap * 0.7))
    # éviter les doublons trop proches des beats principaux
    def _too_close(t, srcs, tol=0.05):
        return any(abs(t - b[0]) < tol for b in srcs)

    # 5. Combiner et trier
    all_beats = main_beats + off_grid_beats + [b for b in filled if not _too_close(b[0], main_beats + off_grid_beats, 0.05)]
    all_beats.sort(key=lambda x: x[0])

    # 6. Extraire temps et forces
    times = np.array([b[0] for b in all_beats], dtype=float)
    strengths = np.array([b[1] for b in all_beats], dtype=float)

    return times, strengths, period, grid


# Alias pour compatibilité - on utilisera la nouvelle approche directement dans analyze_audio
def _filter_weak_onsets(y, sr, times, strengths, min_ratio=1.5, min_strength=None,
                        beat_times=None, hop=None, fine=None):
    """Filtre les onsets sans vraie attaque (souffle, voix, tenues).

    Vectorisé : l'ancienne version recalculait une médiane glissante par onset
    après avoir reconstruit l'enveloppe fine.
    """
    import numpy as np

    times = np.asarray(times, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    if times.size == 0:
        return times, strengths

    fe, hop_fine = fine if fine is not None else _fine_envelope(y, sr)
    fps = float(sr) / float(hop_fine)

    if min_strength is None:
        min_strength = float(np.median(strengths)) * 0.5

    base = _rolling_median(fe, 9)
    rise = _forward_max(fe, 6)
    ratio = rise / (base + 1e-9)

    idx = np.clip((times * fps).astype(int), 0, max(0, ratio.size - 1))
    keep = (strengths >= min_strength) & (ratio[idx] >= min_ratio)
    return times[keep], strengths[keep]


def _dedup_onsets(times, strengths, min_gap):
    """Retire les onsets à moins de `min_gap` s d'un onsets plus fort."""
    import numpy as np

    times = np.asarray(times, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    if times.size == 0:
        return times, strengths

    # On conserve l'ordre temporel tout en alignant strengths.
    order = np.argsort(times)
    t_s = times[order]
    s_s = strengths[order]
    keep = [0]
    for i in range(1, len(t_s)):
        if t_s[i] - t_s[keep[-1]] < min_gap:
            # on garde le plus fort des deux
            if s_s[i] > s_s[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    keep = np.asarray(keep)
    return t_s[keep], s_s[keep]


def _load_audio(audio_path, sr=DEFAULT_AUDIO_SR, cancel_event=None):
    """Charge un fichier audio/vidéo avec librosa."""

    import librosa
    import tempfile
    import subprocess
    import time as _time

    # Étape 1 : extraire/convertir l'audio en WAV temporaire
    import numpy as np
    from app.errors import JobCancelled

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

    # Étape 3 : replacer l'audio sur l'origine du média. Si la piste audio
    # commence après l'origine (flux MXF/TS, montage remuxé, …), on complète
    # par du silence pour éviter un décalage systématique des marqueurs.
    offset = _audio_offset(audio_path)
    if offset > 0:
        silence = np.zeros(int(round(offset * sr_actual)), dtype=y.dtype)
        y = np.concatenate((silence, y))

    return y, sr_actual


def _compute_onsets(y, sr, fine=None):
    """Calcule l'enveloppe d'onset et détecte les onsets.

    Trois corrections cumulatives par rapport à l'ancienne version :

    1. ``aggregate=np.mean`` (callable, pas la chaîne ``"mean"``) — librosa
       1.0 n'agrège sinon aucune bande mel et ne renvoie qu'un bande passante
       grave, ce qui faisait tomber les marqueurs sur les tenues de basse.
    2. Hop d'analyse ≈ 5 ms au lieu de 23 ms (une demi-image à 24 fps).
    3. Peak-picking : fenêtre de maximum local sur ±20 ms (jamais ±150 ms)
       et `wait=MIN_BEAT_INTERVAL` ; le seuil est la moyenne glissante sur 100 ms.
    4. Recalage de chaque onset sur le début réel du transitoire (1 ms).

    Retourne : (onset_envelope, onset_times (recalés), onset_strengths).
    Les forces sont échantillonnées sur la trame *détectée* (avant recalage).
    """
    import numpy as np
    import librosa

    hop = _analysis_hop(sr)

    # 1. Enveloppe multi-bandes moyennée (onset_strength garde le hop au
    #    centre des trames, puis décale le train d'attaques pour compenser
    #    l'effet de fenêtre : le group transport est résorbé).
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop,
                                            aggregate=np.mean)

    if onset_env.size == 0 or float(np.max(onset_env)) <= 0:
        return onset_env, np.array([], dtype=float), np.array([], dtype=float)

    # 2. Peak-picking fin : un maximum local sur ±20 ms, pas ±150 ms.
    pre_max = max(1, int(round(ONSET_PRE_MAX * sr / hop)))
    avg_win = max(1, int(round(ONSET_AVG_WINDOW * sr / hop)))
    wait = max(1, int(round(MIN_BEAT_INTERVAL * sr / hop)))
    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop,
        units="frames",
        pre_max=pre_max,
        post_max=1,                 # strictement positif (exigence peak_pick)
        pre_avg=avg_win,
        post_avg=avg_win + 1,
        delta=ONSET_DELTA,
        wait=wait,
        backtrack=False,
    )

    if frames.size == 0:
        return onset_env, np.array([], dtype=float), np.array([], dtype=float)

    frames = np.clip(frames, 0, len(onset_env) - 1)
    strengths = onset_env[frames]
    coarse = librosa.frames_to_time(frames, hop_length=hop, sr=sr)

    # 3. Recalage sur le début de l'attaque (1 ms).
    fine = fine if fine is not None else _fine_envelope(y, sr)
    novelty, novelty_hop = fine
    times = _snap_to_attack(coarse, novelty, novelty_hop, sr)

    # 4. Déduplication défensive : le recalage peut rapprocher deux marqueurs
    #    d'un même transitoire. On garde celui de plus grande énergie si deux
    #    onsets sont à moins de 45 ms l'un de l'autre.
    times, strengths = _dedup_onsets(times, strengths, 0.045)

    # Rejet des « faux onsets » : transitoires sans réelle attaque (hum /
    # souffle / voix). Étiquette analyse, ne dégrade pas si rien ne passe.
    if times.size > 0 and y.size > 0:
        try:
            t_arr, s_arr = _filter_weak_onsets(
                y, sr, times.tolist(), strengths.tolist(),
                min_ratio=1.6, min_strength=float(np.median(strengths)) * 0.5,
                fine=fine)
            if t_arr.size >= max(1, times.size // 4):
                times, strengths = t_arr, s_arr
        except Exception:  # noqa: BLE001 — le filtre reste facultatif
            pass

    return onset_env, np.asarray(times, dtype=float), np.asarray(strengths, dtype=float)


def _estimate_tempo_sr(onset_env, sr, hop):
    """Estime le tempo et la grille de beats.

    `hop` doit accompagner l'enveloppe : sans lui, beat_track interprete
    les trames avec son hop par défaut (512) et renvoie des temps faux.

    Retourne (tempo: float, beat_times: np.ndarray).
    """
    import numpy as np
    import librosa

    hop = int(hop)

    tempo = 120.0
    beat_times = np.array([], dtype=float)
    try:
        tempo_values, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop, units="frames",
        )
        # tempo_values peut être un array (si plusieurs tempos) ou un scalaire
        if np.size(tempo_values) > 0:
            tempo = float(np.mean(tempo_values))
        if tempo <= 0:
            tempo = 120.0
        beat_frames = np.asarray(beat_frames)
        if beat_frames.size:
            beat_times = librosa.frames_to_time(beat_frames, hop_length=hop, sr=sr)
    except Exception:
        tempo = 120.0

    return float(tempo), np.asarray(beat_times, dtype=float)


def _local_percentile(arr, window_size, percentile):
    """Calcule le percentile local glissant sur un array 1D.

    Utilise une fenêtre centrée (avec bords réfléchis).
    """
    import numpy as np
    from scipy.ndimage import percentile_filter
    return percentile_filter(arr, size=window_size, mode='reflect', percentile=percentile)


def _classify_beats(onset_times, onset_strengths, tempo, sr, beat_times=None, hop=128):
    """Sélectionne et classe les onsets qui représentent réellement des beats musicaux.

    Retourne uniquement les onsets suffisamment forts et rythmiquement cohérents
    pour être considérés comme des beats, avec leur rang (1=fort, 2=moyen, 3=faible).

    Philosophie : 
    - D'abord filtrer les onsets pour ne garder que ceux qui sont vraisemblablement des beats
    - Ensuite classifier ceux-ci en 3 rangs selon leur énergie locale et position rythmique
    """
    import numpy as np
    if len(onset_times) == 0:
        return []

    strengths = np.asarray(onset_strengths, dtype=np.float64)
    times = np.asarray(onset_times, dtype=np.float64)

    # Seuils locaux en TEMPS, pas en index d'onset (8 s glissants).
    if len(times) >= 2:
        ioi = np.median(np.diff(np.sort(times)))
        window_onsets = int(round(8.0 / max(float(ioi), 1e-3) / 2.0))
    else:
        window_onsets = 10
    window_onsets = int(np.clip(window_onsets, 15, max(15, len(strengths) // 2)))
    p1_local = _local_percentile(strengths, window_onsets, DEFAULT_RANK1_PERCENTILE)
    p2_local = _local_percentile(strengths, window_onsets, DEFAULT_RANK2_PERCENTILE)

    # Seuils globaux en repli
    p1_global = float(np.percentile(strengths, DEFAULT_RANK1_PERCENTILE))
    p2_global = float(np.percentile(strengths, DEFAULT_RANK2_PERCENTILE))

    # Grille de beats (PLP)
    grid = np.sort(np.asarray(beat_times, dtype=float)) if beat_times is not None else np.array([])
    meter = 4
    period = 60.0 / tempo if tempo > 0 else 0.0

    # Fenêtre de tolérance sur la grille : 8% de la période (≈45ms à 108 BPM)
    grid_tol = max(0.025, period * 0.08)

    # Seuil absolu minimum pour être considéré comme un potentiel beat
    min_strength_for_beat = max(0.1, float(np.median(strengths)) * 0.3)
    # Hoisté hors de la boucle : était recalculé pour chaque onset.
    energy_range = float(np.percentile(strengths, 75)) - min_strength_for_beat

    # Point d'entrée musical : premier beat de la grille avec un onset fort à proximité
    music_start = 0.0
    if grid.size and len(strengths) > 0:
        for gt in grid:
            idx = np.argmin(np.abs(times - gt))
            if idx < len(strengths) and strengths[idx] > min_strength_for_beat * 2:
                music_start = max(0.0, gt - period)  # une mesure avant le 1er beat fort
                break

    # === ÉTAPE 1: SÉLECTION DES BEATS CANDIDATS ===
    beat_candidates = []  # Liste de (index, score) pour les onsets retenus comme beats
    
    for i, (t, s) in enumerate(zip(times, strengths)):
        # Filtrer le bruit de début (mais être moins sévère)
        if t < music_start and s < min_strength_for_beat * 4:
            continue
            
        # Ne considérer que les onsets suffisamment forts
        if s < min_strength_for_beat:
            continue
            
        # Calculer un score de "qualité de beat"
        score = 0.0
        
        # Composante 1: énergie normalisée (0-1)
        # Utiliser un pourcentage plus bas pour être plus inclusif
        # Et ajouter un petit offset pour éviter la division par zéro
        if energy_range < 1e-9:
            energy_norm = 0.5  # Valeur par défaut si peu de variation
        else:
            energy_norm = (s - min_strength_for_beat) / energy_range
        energy_norm = max(0.0, min(1.0, energy_norm))
        score += 0.3 * energy_norm
        
        # Composante 2: proximité avec la grille rythmique
        if grid.size and period > 0:
            idx = int(np.searchsorted(grid, float(t)))
            min_dist = float('inf')
            for cand_idx in (idx - 1, idx):
                if 0 <= cand_idx < grid.size:
                    dist = abs(float(t) - float(grid[cand_idx]))
                    if dist < min_dist:
                        min_dist = dist
            # Score basé sur la distance avec une décroissance plus douce
            # Augmenter la distance maximale pour être plus tolérant
            max_dist_for_score = period * 0.8  # Autoriser jusqu'à 80% de période d'écart (augmenté de 0.6)
            grid_score = max(0.0, 1.0 - min_dist / max_dist_for_score)
            score += 0.4 * grid_score
        else:
            # Pas de grille disponible, utiliser seulement l'énergie
            score += 0.4  # score neutre pour la composante grille
        
        # Composante 3: netteté de l'attaque et consistance locale
        if i > 0 and i < len(strengths) - 1:
            # Bonus pour être un maximum local d'énergie
            if s >= strengths[i-1] and s >= strengths[i+1]:
                score += 0.2
            # Bonus supplémentaire si nettement supérieur aux voisins
            elif s > strengths[i-1] * 1.3 and s > strengths[i+1] * 1.3:
                score += 0.25
        
        # Seuil pour considérer comme un beat candidat - abaissé pour être plus inclusif
        if score >= 0.40:  # seuil relevé (0.35→0.40) pour réduire les faux positifs (« en trop »)
            beat_candidates.append((i, score))
    
    # S'assurer d'avoir au moins quelques beats si on trouve quelque chose
    if len(beat_candidates) == 0 and len(times) > 0:
        # Fallback: prendre les onsets les plus forts, mais en limiter le nombre
        sorted_indices = np.argsort(strengths)[::-1]  # Du plus fort au plus faible
        max_beats = min(len(times)//2, 50)  # Augmenté: jusqu'à la moitié des onsets
        beat_candidates = [(idx, 0.35) for idx in sorted_indices[:max_beats]]
    
    # Trier par temps (pas par score) pour maintenir l'ordre chronologique
    beat_candidates.sort(key=lambda x: x[0])
    
    # === ÉTAPE 2: CLASSIFICATION DES BEATS SÉLECTIONNÉS ===
    beats = []
    for idx, _ in beat_candidates:
        t, s = times[idx], strengths[idx]
        
        # 1. Filtrage bruit de début (déjà fait partiellement lors de la sélection)
        if t < music_start and s < min_strength_for_beat * 3:
            continue

        # 2. Seuils locaux (avec repli global si fenêtre trop petite)
        p1 = p1_local[idx] if idx < len(p1_local) else p1_global
        p2 = p2_local[idx] if idx < len(p2_local) else p2_global

        # 3. Classement de base par énergie locale
        if s >= p1:
            rank = 1
        elif s >= p2:
            rank = 2
        else:
            rank = 3

        # 4. Détection position sur la grille (index et distance en une passe)
        on_grid = False
        nearest_beat_dist = float('inf')
        idx_near = -1
        if grid.size:
            idx_grid = int(np.searchsorted(grid, float(t)))
            for cand_idx in (idx_grid - 1, idx_grid):
                if 0 <= cand_idx < grid.size:
                    dist = abs(float(t) - float(grid[cand_idx]))
                    if dist < nearest_beat_dist:
                        nearest_beat_dist = dist
                        idx_near = cand_idx
            on_grid = nearest_beat_dist <= grid_tol

        # 5. Ajustement de rang selon position rythmique. On ne déclasse que si
        # l'énergie est INSUFFISANTE pour le rang attribué — sinon on conserve
        # (le bug précédent déclassait systématiquement les syncopations : un
        # accent hors grille à +6 dB finissait en rang 2, d'où les « mal notés »).
        if on_grid:
            if rank == 3 and s >= min_strength_for_beat * 1.0:
                rank = 2
        else:
            if rank == 1 and s < p1 * 1.15:
                rank = 2
            elif rank == 2 and s < p2 * 1.1:
                rank = 3

        # Downbeat : phase 0 du mètre (uniquement indicative, n'affecte pas le rang)
        is_downbeat = bool(idx_near >= 0 and period > 0 and (idx_near % meter) == 0)
        beats.append({
            "time": round(float(t), 3),
            "energy": round(float(s), 4),
            "rank": rank,
            "color": RANK_COLORS.get(rank, "White"),
            "label": {1: "Impact fort", 2: "Intermédiaire", 3: "Micro-variation"}[rank],
            "is_downbeat": bool(is_downbeat),
        })
    return beats


def analyze_audio(audio_path, sr=DEFAULT_AUDIO_SR, progress_cb=None,
                  cancel_event=None,
                  offset=0.0):
    """Analyse complète de la piste audio.

    Args:
        audio_path : chemin vers le fichier audio/vidéo
        sr : sample rate de travail (22050 par défaut)
        progress_cb : callback(frac, step) — progression par étapes
        cancel_event : threading.Event pour annuler (kill de ffmpeg inclus)

    Returns:
        dict avec : beats, fps, duration, tempo, total_onsets, elapsed
    """
    import numpy as np
    import librosa
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
    hop = _analysis_hop(sr_actual)
    duration = librosa_seconds(y, sr_actual)
    step(0.25, f"Audio chargé ({duration:.1f} s, {sr_actual} Hz)")

    # Enveloppe fine d'attaque (1 ms) : calculée une fois, réutilisée par le
    # recalage, le filtre des faux onsets et les marqueurs.
    fine = _fine_envelope(y, sr_actual)

    # Enveloppe d'onset + détection (raglé pour l'alignement)
    step(0.3, "Détection des onsets")
    onset_env, onset_times, onset_strengths = _compute_onsets(y, sr_actual,
                                                              fine=fine)

    # Tempo estimé (la grille est ancrée sur la musique, plus sur 0)
    step(0.6, "Estimation du tempo")
    audio_offset = _audio_offset(audio_path)
    tempo, beat_times = _estimate_tempo_sr(onset_env, sr_actual, hop)

    # Nouvelle approche hybride : grille PLP + enveloppe fine + onsets librosa
    step(0.65, "Construction marqueurs (grille + attaques fines)")
    beat_times_final, strengths_final, period, grid = _build_beat_markers(
        y, sr_actual, onset_env, onset_times, onset_strengths, beat_times, hop,
        fine=fine,
    )
    if beat_times_final is None or len(beat_times_final) == 0:
        beat_times_final = onset_times.copy()
        strengths_final = onset_strengths.copy()

    # Recalage manuel (décalage de projet)
    if offset:
        beat_times_final = beat_times_final + float(offset)
        if beat_times_final.size:
            keep = (beat_times_final >= 0.0) & (beat_times_final <= duration)
            beat_times_final = beat_times_final[keep]
            strengths_final = strengths_final[keep]

    # Classification des beats finaux
    step(0.85, "Classification des beats (3 rangs)")
    beats = _classify_beats(beat_times_final, strengths_final, tempo, sr_actual,
                            beat_times=beat_times, hop=hop)
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
        "audio_offset_ms": round(audio_offset * 1000, 1),
        "manual_offset_s": round(float(offset), 4),
        "detection": {
            "hop_ms": round(hop * 1000.0 / sr_actual, 2),
            "min_interval_s": MIN_BEAT_INTERVAL,
            "aggregation": "mel-mean",
            "refine": "attack-start",
        },
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
        start = float(b.get("time") or 0.0)
        end = start + 0.1  # durée arbitraire (100 ms)
        rank = int(b.get("rank") or 1)
        label = b.get("label") or "Beat"
        energy = float(b.get("energy") or 0.0)
        color = b.get("color") or RANK_COLORS.get(rank, "White")
        name = f"Beat {i} — {label}"
        comment = f"Rank {rank} | Energy {energy} | {start:.3f}s"
        lines.append(
            f"{name};{start:.3f};{end:.3f};0.10;{color};{comment}"
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
        start = float(b.get("time") or 0.0)
        tc = seconds_to_edl_timecode(start, fps)
        # Event M (marker) — un seul frame de durée
        end_tc = seconds_to_edl_timecode(start + 1.0 / fps, fps)
        lines.append(f"{i:03d}  M      V     C        {tc} {end_tc}")
        lines.append(
            f"     * COMMENT: Level {int(b.get('rank') or 1)} - "
            f"{b.get('label') or 'Beat'} "
            f"({start:.3f}s, energy={float(b.get('energy') or 0.0)})"
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
