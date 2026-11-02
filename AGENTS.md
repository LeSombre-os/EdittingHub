# EdittingHub — AGENTS.md

## Vue d'ensemble
Hub local 100% offline (aucun cloud, aucune télémétrie) regroupant des outils de
montage vidéo existants et de nouveaux modules de traitement local, sous forme
d'une app web locale (FastAPI backend + interface statique).

## Architecture
```
EdittingHub/
├── run.sh                    # Lanceur bash : démarre le serveur + ouvre le navigateur
├── stop.sh                   # Arrêt propre (évite le pkill qui tue le shell)
├── main.py                   # Point d'entrée Python (alternative à run.sh)
├── tests/smoke.py            # 54+ tests de fumée (script DaVinci, FCPXML, exports, API, rythme)
├── requirements.txt          # Dépendances Python
├── app/
│   ├── config.py             # Configuration : chemins, constantes, vérif outils
│   ├── utils.py              # Utilitaires : timecode, formatage, parsing
│   ├── server.py             # Application FastAPI + routes API
│   ├── davinci.py            # Script marqueurs Resolve + installation menu Scripts
│   ├── fcpxml.py             # FCPXML 1.9 (marqueurs beats, découpe de plans, montage rythme)
│   ├── media.py              # ffprobe : durée, codecs, dimensions
│   ├── procs.py              # Sous-processus + killpg (annulation réelle)
│   ├── estimates.py          # Estimations de durée calibrées
│   ├── errors.py             # Exceptions métier (JobCancelled)
│   ├── services/
│   │   ├── convert.py        # Hub Conversion  → subprocess vers ResolveConvert/convert.py
│   │   ├── subtitles.py      # Hub Sous-titrage → subprocess vers ResolveSubs/transcrire.py
│   │   ├── scene_detect.py   # Découpage plans → PySceneDetect (scenedetect 0.7.x)
│   │   ├── audio_beats.py    # Analyse audio → Librosa 1.0
│   │   └── rhythm.py         # Montage au rythme → coupes séquentielles, plan, exports
│   └── static/
│       ├── index.html        # Interface web (CSS inline + Vanilla JS, 0 CDN)
│       └── icon.svg          # Icône du projet (en-tête web, favicon, menu apps)
```

## Démarrage
```bash
cd EdittingHub
./run.sh              # ou : python3 main.py
```
→ ouvre http://127.0.0.1:8005 dans le navigateur par défaut.

## Modules

### 1. Hub de Conversion
- Wrappe `ResolveConvert/convert.py` par subprocess (`python3 convert.py`)
- Paramètres : target (resolve/tiktok), preset (lb/sq/hq), keep_4k, force, jobs
- Sortie : `CONVERTIS/` (Resolve) ou `TIKTOK/` (mp4)

### 2. Hub de Sous-titrage
- Wrappe `ResolveSubs/transcrire.py` par subprocess
- Utilise le venv `resolvesubs` (faster-whisper) ou python3 système
- Paramètres : language, model, device, offset, mode, music, prompt
- Sortie : `SOUS_TITRES/<nom>.srt`

### 3. Découpage Automatique de Plans
- Utilise `scenedetect` (PySceneDetect) directement (pas de subprocess)
- API 0.7.x : `AVAILABLE_BACKENDS['opencv']` pour ouvrir la vidéo
- Export : EDL (CMX 3600), XML (FCPXML), CSV

### 4. Analyseur de Beats Audio
- Utilise `librosa` directement (pas de subprocess)
- Extraction d'onsets → classification 3 rangs par percentile d'énergie
- Filtre des rangs côté serveur (`ranks: list[int]`) et côté UI (cases 1/2/3)
- Export : CSV, EDL (markers M), FCPXML, script Python DaVinci

### 5. Montage au Rythme (auto-cut beats)
- Nouveau moteur `app/services/rhythm.py` : `selected_beats`, `cut_points`, `build_cut_plan` (min/max, subdivisions, bouclage, `video_start`, troncature), `export_rhythm_edl/csv`, `export_rhythm`, `probe_sources`.
- `app/fcpxml.py` : helper `_append_cut_clips` + `rhythm_document` (vidéo + musique liée `lane="-1"`, option `include_audio`).
- Serveur : `RhythmScanRequest`, `RhythmRequest`, `RhythmExportRequest`, `_rhythm_sources`, `_run_rhythm_job`, routes `POST /api/rhythm/scan`, `/api/rhythm/start`, `/api/rhythm/export`.
- UI : 5e onglet `panel-rhythm`, CSS `--rhy`, sources vidéo/musique + explorateur intégré, réglages min/max/fps/départ, rangs 1/2, bouclage, musique, boutons Vérifier/Générer/Arrêter, stats + table + exports.
- Export FCPXML : plans vidéo + musique sur piste liée (`lane="-1"`), importable par Resolve.
- 100 % local, aucune inférence LLM, aucun upload.

### 5. Marqueurs DaVinci Resolve (`app/davinci.py`)
- **L'import FCPXML/EDL ne restitue PAS les marqueurs dans Resolve Free Linux.**
  Ne pas promettre le contraire : c'est une limite de l'importeur, pas de l'export.
- Voie fiable = API Python **interne** (`resolve` fourni par la console / le menu
  *Workspace → Scripts*). Le scripting externe renvoie `resolve = None` ici.
- `build_script()` génère un `.py` **autonome** (marqueurs embarqués, encodage
  compact `t|rang|nom` sur **une seule ligne**).
- `install_script()` écrit le fichier dans
  `~/.local/share/DaVinciResolve/Fusion/Scripts/Utility/EdittingHub_Beat_Markers.py`.
  Resolve n'énumère ce dossier qu'au démarrage → **redémarrage la 1re fois**,
  puis le fichier est relu à chaque clic (pas besoin de redémarrer).
- Endpoint : `POST /api/audio/davinci/install` (`job_id`, `fps`, `ranks`, `file_index`).

## Conventions
- 100% local : zéro import de bibliothèques externes dans le frontend
- CSS inline via variables `--bg`, `--card`, `--acc`, etc.
- Vanilla JS, pas de framework (React/Vue)
- Backend : FastAPI + uvicorn (threadpool pour les jobs background)
- Jobs asynchrones : thread + dict en mémoire (`JOBS`)
- Annulation propre via `threading.Event`

## Tests rapides
```bash
curl http://127.0.0.1:8005/api/ping
curl http://127.0.0.1:8005/api/config
curl "http://127.0.0.1:8005/api/ls?path=/media"

# Analyse audio réelle (133 beats sur fragment-trailer.mp4)
curl -X POST http://127.0.0.1:8005/api/audio/analyze \
     -H 'Content-Type: application/json' \
     -d '{"paths":["/media/lesombre/2A14-B525/SSD_E/Opencode/_externes/test_vidéo/fragment-trailer.mp4"]}'

# Exports (rangs 1+2 = 101 marqueurs)
curl -X POST http://127.0.0.1:8005/api/audio/export \
     -H 'Content-Type: application/json' \
     -d '{"job_id":1,"fmt":"davinci_script","fps":24,"ranks":[1,2]}'

# Installation dans le menu Scripts de Resolve
curl -X POST http://127.0.0.1:8005/api/audio/davinci/install \
     -H 'Content-Type: application/json' \
     -d '{"job_id":1,"fps":24,"ranks":[1,2]}'
```

Tests de fumée (à lancer après toute modif backend) :

```bash
~/.venvs/edittinghub/bin/python tests/smoke.py        # unitaire, < 1 s (54 tests)
~/.venvs/edittinghub/bin/python tests/smoke.py --api  # + HTTP (serveur sur 8005, 76 tests)
```

Vérifier un script généré sans Resolve :

```bash
python3 -m py_compile ~/.local/share/DaVinciResolve/Fusion/Scripts/Utility/EdittingHub_Beat_Markers.py
grep -n '^[[:space:]]\+[^#]' <fichier>.py   # doit être vide (aucune ligne indentée)
```

## Vérifications avant de livrer une modif UI

1. **IDs uniques** : `grep -c 'id="browser"' app/static/index.html` doit renvoyer 0
   (un conteneur par onglet : `browser-conv`, `browser-sub`, `browser-scene`,
   `browser-audio`, `browser-rhythm`).
2. **Syntaxe JS** : extraire le `<script>` et `node --check`.
3. **Cohérence** : chaque `$('id')` doit exister ; chaque handler `onclick=` doit
   être défini **et** exporté via `Object.assign(window, …)` ; chaque clé
   `t.<x>` du frontend doit exister dans `tools` de `/api/config`
   (attention : `subtitles_script`, pas `subtitle_script`).
4. **Clics réels** : les onglets doivent avoir un `addEventListener('click', …)`
   — un `onclick` inline absent = onglets morts. Test headless possible :
   copier `index.html` en `app/static/__test_*.html` avec un script de test,
   puis `brave-browser --headless --dump-dom http://127.0.0.1:8005/static/__test_*.html`
   (le serveur ne monte que `/static`), et supprimer le fichier ensuite.
5. **E2E backend** : scan conversion + scan sous-titrage, job scènes + exports
   EDL/XML/CSV, job audio + exports CSV/EDL/XML/`davinci_script`, job rythme
   (scan + start + exports FCPXML/EDL/CSV), puis
   `POST /api/audio/davinci/install` → vérifier que le `.py` écrit dans
   `Fusion/Scripts/Utility/` passe `python3 -m py_compile` et ne contient
   aucune ligne indentée.

## Pièges connus
- `subtitles.py` parse la sortie de `transcrire.py` : la ligne réelle est
  « Terminé en 0:03 : 1 **transcrits**, 0 déjà OK, 0 échecs. » (regex alignée sur
  ce texte exact).
- Un trailer purement instrumental fait échouer le sous-titrage avec
  « aucune parole détectée » : comportement normal, relancer en mode musique.
- `run.sh` attend que `/api/ping` réponde (max 20 s) avant d'ouvrir le navigateur ;
  ne pas revenir à un `sleep` fixe.
- **Ne jamais importer `DaVinciResolveScript` / charger `fusionscript.so` dans un
  script exécuté *dans* Resolve** : double chargement → `SIGSEGV` (Signal 11).
  Le script généré réutilise `resolve` via `globals()` → `bmd` → `sys.modules['fusionscript']`.
- **`Timeline.AddMarker()` attend un numéro de frame entier**, jamais un timecode
  (`int(seconds * fps + 0.5)`). Un type non entier = crash.
- Chaque instruction du script généré doit tenir sur **une seule ligne** (pas de
  bloc indenté) pour rester valide si la console exécute le collage ligne à ligne.
- Les `print()` du script sont forcés en ASCII (`encode('ascii','replace')`) :
  la console Resolve n'est pas garantie UTF-8.
- **Ne pas toucher à `config.dat` / `.config.data`** de Resolve (surtout GPU) :
  Resolve réécrit ces fichiers à sa fermeture. Le mode scripting reste à `1`
  (console + menu Scripts fonctionnent sans mode externe).
- `pkill -f "uvicorn app.server:app"` tue aussi le shell qui contient la commande :
  utiliser `./stop.sh` (prévu pour ça) ou un motif `[u]vicorn`.

## Nouveautés (v1.2.0)
- **Annulation réelle** : chaque job de conversion/sous-titrage tue le groupe
  de processus (SIGTERM → SIGKILL via `killpg`) incluant les enfants
  (ffmpeg/whisper). Le bouton « Arrêt » coupe les processus au fond, pas
  seulement le parent. Voir `app/procs.py` / `server.py`.
- **Sélecteur natif** : les boutons « Parcourir » ouvrent `zenity --file-selection`
  (séparateur `\n`, filtres par extension, dossier multiple). Si zenity ou
  l'affichage manque, bascule automatiquement sur l'explorateur intégré.
- **Estimations calibrées** : ratios `s de traitement / s de média` par module
  (convert `target:preset`, subtitles `model:device`, scenes `dsN`, audio),
  lissés par moyenne pondérée 0.5 et stockés dans `outputs/estimates.json`.
  L'UI affiche la durée estimée avant le lancement et l'ETA en direct.
- **Export ZIP (batch)** : détection/scène/audio sur plusieurs fichiers → ZIP
  contenant les exports (EDL/XML/CSV ou XML/CSV/EDL selon le format choisi).
- **FCPXML 1.9 correct** : structure `library > event > project > sequence > spine > asset-clip > marker`
  (pas de markers directs sur la séquence, asset avec `media-rep` + `src`).
  Importable par DaVinci Resolve (testé : 22 plans, 133 beats sur 2 fichiers).
- **Interface mise à jour** : durée estimée, ETA live, boutons « Dossier » et
  « Parcourir » natifs par onglet, arrêt « Traitement interrompu », exports
  ZIP par lot, fichiers sélectionnables parmi le lot.

## Nouveautés (v1.3.0)
- **Marqueurs DaVinci fiables** : bouton **⚡ Envoyer à DaVinci** → écrit
  `EdittingHub_Beat_Markers.py` dans `Fusion/Scripts/Utility/`, puis
  *Workspace → Scripts → EdittingHub Beat Markers*. Remplace la voie
  « terminal externe » qui renvoyait `resolve = None`.
- **Script durci** : zéro import, frames entiers, une instruction par ligne,
  sorties ASCII, `CLEAR_FIRST` optionnel. Fini les `SIGSEGV` de la console.
- **Téléchargement du `.py`** conservé (méthode console / installation manuelle).
- **Aide intégrée réécrite** en français : les 2 méthodes, le résultat attendu,
  les 4 erreurs courantes et leur solution.
- **README complet** (installation, modules, API, architecture, dépannage, limites).
- **Nettoyage UI** : suppression du texte d'aide dupliqué et des libellés
  parasites (`(CancellationToken)`, `showDavinciRunHelp`).

## Nouveautés (v1.4.0)
- **Module Montage au rythme (auto-cut beats)** : 5e onglet UI, découpe
  séquentielle de la vidéo sur les beats (rangs 1/2), plans min/max, bouclage
  optionnel, départ vidéo, exports FCPXML (piste musique liée `lane="-1"`),
  EDL (4 timecodes), CSV.
- **API rythme** : `/api/rhythm/scan` (vérification sources), `/api/rhythm/start`
  (job analyse + plan), `/api/rhythm/export` (FCPXML/EDL/CSV).
- **Moteur `rhythm.py`** : sélection beats par rangs, points de coupe, plan de
  montage complet (bouclage, troncature, stats), exports.
- **FCPXML étendu** : `rhythm_document()` avec musique sur piste liée,
  `_append_cut_clips()` factorisé.
- **Tests** : `test_rhythm()` + `test_api_rhythm()` (scan, start, exports, erreurs
  400/404), 76 tests totaux (`--api`).
- **UI** : onglet `rhythm`, variables CSS `--rhy/--rhy2`, explorateur natif,
  cases rangs 1/2 (3 décoché), bouton boucle, piste musique optionnelle.
- **Version** : 1.4.0 (serveur + footer UI).
