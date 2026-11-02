# EdittingHub

Studio de montage **100 % local** (aucun cloud, aucune télémétrie, aucune IA
distante) qui réunit quatre outils autour d'une même interface web :

| Module | Ce qu'il fait | Techno |
|---|---|---|
| **Conversion** | Convertit les rushes vers un codec montable dans DaVinci Resolve (DNxHR) ou vers du MP4 vertical (TikTok) | `ffmpeg` via `ResolveConvert/convert.py` |
| **Sous-titrage** | Transcrit l'audio en `.srt` (faster-whisper, modèles locaux) | `ResolveSubs/transcrire.py` |
| **Plans** | Découpe une vidéo en plans et exporte EDL / FCPXML / CSV | PySceneDetect + OpenCV |
| **Audio** | Détecte les *beats*, les classe en 3 rangs d'énergie et les pose en **marqueurs dans DaVinci Resolve** | Librosa |

Interface : un seul fichier HTML (CSS + Vanilla JS, zéro CDN, zéro framework).
Backend : FastAPI + uvicorn, jobs asynchrones annulables.

---

## Sommaire

1. [Démarrage rapide](#démarrage-rapide)
2. [Les quatre modules](#les-quatre-modules)
3. [Marqueurs de beats dans DaVinci Resolve](#marqueurs-de-beats-dans-davinci-resolve)
4. [API HTTP](#api-http)
5. [Architecture du code](#architecture-du-code)
6. [Dépannage](#dépannage)
7. [Limites connues](#limites-connues)

---

## Démarrage rapide

### Prérequis

- Linux (testé sur Zorin OS 17 / Ubuntu 24.04)
- Python 3.11+
- `ffmpeg` et `ffprobe` dans le `PATH`
- DaVinci Resolve (gratuit) pour l'usage des marqueurs
- Facultatif : `zenity` pour les sélecteurs de fichiers natifs

```bash
sudo apt install ffmpeg zenity
```

### Installation

```bash
cd EdittingHub
python3 -m venv ~/.venvs/edittinghub
~/.venvs/edittinghub/bin/pip install -r requirements.txt
```

Le projet cherche automatiquement l'interpréteur dans `~/.venvs/edittinghub/bin/python`
et retombe sur `python3` s'il n'existe pas.

Les modules **Conversion** et **Sous-titrage** appellent des scripts voisins
(`../ResolveConvert/convert.py`, `../ResolveSubs/transcrire.py`). Le chemin est
déduit du dossier parent par défaut ; s'ils sont ailleurs :

```bash
export EDITTINGHUB_WORKSPACE=/chemin/du/workspace
```

### Lancement

```bash
./run.sh          # démarre le serveur et ouvre le navigateur
# ou
python3 main.py   # équivalent en Python pur
```

L'interface est servie sur **<http://127.0.0.1:8005>**.
Le serveur écoute uniquement sur `127.0.0.1` : rien n'est exposé au réseau.

Pour arrêter : `pkill -f "uvicorn app.server:app"`.

---

## Les quatre modules

### 1. Conversion

Deux cibles :

- **Resolve** — DNxHR (`.mov`), presets `lb` / `sq` / `hq`, option « conserver la 4K ».
  Indispensable sous Linux : Resolve Free n'ouvre pas le H.264 (`avc1`) de
  beaucoup de rushes.
- **TikTok** — MP4 vertical prêt à publier.

Sorties : `CONVERTIS/` et `TIKTOK/` (dans `ResolveConvert/`).

### 2. Sous-titrage

Transcription locale (faster-whisper). Paramètres exposés : langue, modèle,
périphérique (`cpu` / `cuda`), décalage, mode (`parole` / `musique`), prompt.

Sortie : `SOUS_TITRES/<nom>.srt`.

> Un trailer purement instrumental échoue volontairement avec
> « aucune parole détectée » : relancez en mode `musique`.

### 3. Découpage de plans

Détection des changements de plan (seuil réglable, durée minimale en frames).
Export **EDL (CMX 3600)**, **FCPXML 1.9** ou **CSV**, par fichier ou en ZIP
pour un lot.

### 4. Analyse audio

Détection d'onsets par Librosa, puis classification en trois rangs selon
l'énergie (percentiles 66 / 33) :

| Rang | Signification | Couleur du marqueur |
|---|---|---|
| 1 | Impact fort (kick, snare, drop) | Rouge |
| 2 | Intermédiaire | Jaune |
| 3 | Micro-variations (décoché par défaut) | Bleu |

Chaque rang est filtrable à l'export via les cases à cocher.

---

## Marqueurs de beats dans DaVinci Resolve

### Pourquoi un script et pas un import ?

L'export FCPXML 1.9 produit par EdittingHub est **valide** (structure
`library > event > project > sequence > spine > asset-clip > marker`), mais
**DaVinci Resolve Free sous Linux n'importe pas les `<marker>`** : la timeline
arrive avec l'audio, sans aucun marqueur. Même constat pour l'EDL. Ce n'est pas
un bug de l'export, c'est une limite de l'importeur de la version gratuite.

Le seul canal fiable est l'**API Python interne** de Resolve — celle qui fournit
l'objet `resolve` dans la console et dans les scripts du menu *Workspace → Scripts*.

### Méthode 1 — « Envoyer à DaVinci » (recommandée)

1. Analysez la musique dans l'onglet **Audio**, cochez les rangs voulus.
2. Cliquez **⚡ Envoyer à DaVinci**.
   Le script est écrit dans :

   ```
   ~/.local/share/DaVinciResolve/Fusion/Scripts/Utility/EdittingHub_Beat_Markers.py
   ```

3. Dans Resolve : **Workspace → Scripts → EdittingHub Beat Markers**.
4. Ouvrez la timeline qui doit recevoir la musique, puis relancez le script.

> **Première installation uniquement** : Resolve n'énumère le dossier `Scripts`
> qu'au démarrage, il faut donc le redémarrer une fois. Ensuite le fichier est
> relu à chaque clic — plus besoin de redémarrer, même après une nouvelle analyse.

### Méthode 2 — console (sans redémarrage)

1. Cliquez **🐍 Télécharger le .py**, ouvrez le fichier dans un éditeur de texte.
2. Dans Resolve : **Workspace → Console** (`Alt+Maj+C`), onglet **Python**
   — surtout **pas** l'onglet *Lua*.
3. Collez tout le contenu, puis `Ctrl+Entrée`.

### Ce que fait le script

```
Timeline  : Ma timeline
FPS       : 24.0
Marqueurs : 101 / 101
OK - marqueurs ajoutes sur la timeline active.
```

- Il lit le framerate réel de la timeline (`timelineFrameRate`).
- Il convertit chaque temps en **numéro de frame entier** :
  `AddMarker()` attend un entier, pas un timecode — un timecode provoque un
  `SIGSEGV` dans Resolve.
- Il **n'importe rien** : il réutilise l'objet `resolve` déjà présent
  (`globals()` → `bmd` → module `fusionscript` déjà chargé). Recharger
  `fusionscript.so` dans le processus de Resolve est précisément ce qui faisait
  planter les anciennes versions.
- Chaque instruction tient sur **une seule ligne** : un collage ligne à ligne
  dans la console reste valide.
- Les sorties écran sont forcées en ASCII, pour éviter un `UnicodeEncodeError`
  si la console n'est pas en UTF-8.

Pour repartir de zéro avant de relancer, passez `CLEAR_FIRST = True` en tête du
script : il supprime d'abord les marqueurs Rouge / Jaune / Bleu.

### Le scripting externe ne marche pas

Lancer le script depuis un terminal :

```bash
export RESOLVE_SCRIPT_API=/opt/resolve/Developer/Scripting
export RESOLVE_SCRIPT_LIB=/opt/resolve/libs/Fusion/fusionscript.so
export PYTHONPATH=$RESOLVE_SCRIPT_API/Modules:$PYTHONPATH
python3 EdittingHub_Beat_Markers.py
```

renvoie `resolve = None` sur cette installation : le mode *External scripting*
n'est pas exposé par la version gratuite. Le script affiche alors un message
explicite et s'arrête proprement — utilisez le menu Scripts ou la console.

---

## API HTTP

Toutes les routes sont locales (`127.0.0.1:8005`).

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/api/ping` | Sonde de disponibilité |
| `GET` | `/api/config` | Outils détectés, valeurs par défaut |
| `GET` | `/api/ls?path=…` | Explorateur de fichiers intégré |
| `POST` | `/api/pick` | Sélecteur natif (zenity) |
| `POST` | `/api/convert/scan` | Liste les fichiers convertibles |
| `POST` | `/api/convert/start` | Lance un job de conversion |
| `POST` | `/api/subtitles/scan` | Liste les médias à sous-titrer |
| `POST` | `/api/subtitles/start` | Lance un job de transcription |
| `POST` | `/api/scenes/scan` | Liste les vidéos à découper |
| `POST` | `/api/scenes/start` | Lance un job de détection de plans |
| `POST` | `/api/scenes/export` | Export EDL / FCPXML / CSV / ZIP |
| `POST` | `/api/audio/analyze` | Lance une analyse de beats |
| `POST` | `/api/audio/export` | Export CSV / EDL / FCPXML / script `.py` |
| `POST` | `/api/audio/davinci/install` | Installe le script dans le menu Scripts |
| `GET` | `/api/job/{id}` | État, progression, résultat d'un job |
| `POST` | `/api/job/{id}/cancel` | Annule le job et tue ses sous-processus |

### Exemples

```bash
curl http://127.0.0.1:8005/api/ping

# Analyse
curl -X POST http://127.0.0.1:8005/api/audio/analyze \
     -H 'Content-Type: application/json' \
     -d '{"paths":["/chemin/musique.mp3"]}'

# Export des rangs 1 et 2 uniquement
curl -X POST http://127.0.0.1:8005/api/audio/export \
     -H 'Content-Type: application/json' \
     -d '{"job_id":1,"fmt":"davinci_script","fps":24,"ranks":[1,2]}'

# Installation dans Resolve
curl -X POST http://127.0.0.1:8005/api/audio/davinci/install \
     -H 'Content-Type: application/json' \
     -d '{"job_id":1,"fps":24,"ranks":[1,2]}'
```

`fmt` accepte : `csv`, `edl`, `xml`, `fcpxml` (alias de `xml`), `davinci_script`.
`ranks` est optionnel ; omis, tous les rangs sont exportés.

---

## Architecture du code

```
EdittingHub/
├── run.sh                     # Lanceur bash (attend /api/ping, puis ouvre le navigateur)
├── main.py                    # Lanceur Python équivalent
├── requirements.txt
├── app/
│   ├── config.py              # Chemins, constantes, détection des outils
│   ├── server.py              # FastAPI : routes, jobs, exports
│   ├── davinci.py             # Génération + installation du script Resolve
│   ├── fcpxml.py              # FCPXML 1.9 (marqueurs, découpe de plans)
│   ├── media.py               # ffprobe : durée, codecs, dimensions
│   ├── procs.py               # Gestion des sous-processus (killpg)
│   ├── estimates.py           # Estimations de durée calibrées
│   ├── errors.py              # Exceptions métier
│   ├── utils.py               # Timecode, tailles, collecte de fichiers
│   ├── services/
│   │   ├── convert.py         # → ResolveConvert/convert.py
│   │   ├── subtitles.py       # → ResolveSubs/transcrire.py
│   │   ├── scene_detect.py    # → PySceneDetect
│   │   └── audio_beats.py     # → Librosa
│   └── static/
│       ├── index.html         # Toute l'UI (CSS + JS inline)
│       └── icon.svg
└── outputs/                   # Scènes, beats, EDL, estimations (non versionné)
```

### Choix techniques

- **Jobs en mémoire** : un `dict` + un `thread` par job, annulation via
  `threading.Event` puis `killpg` (SIGTERM → SIGKILL) pour tuer réellement
  `ffmpeg` / `whisper`.
- **Estimations calibrées** : ratios *secondes de traitement / secondes de média*
  par module, lissés par moyenne pondérée (poids 0.5) dans
  `outputs/estimates.json`.
- **Aucune dépendance frontend** : pas de CDN, pas de build, un seul
  `index.html` lisible hors ligne.
- **Exports texte** : CSV / EDL / FCPXML sont générés en Python pur, sans
  bibliothèque XML tierce.

---

## Dépannage

| Symptôme | Cause | Solution |
|---|---|---|
| `Le serveur n'a pas démarré` | port 8005 occupé, dépendance manquante | voir `/tmp/edittinghub.log` |
| Boutons « Parcourir » sans effet | `zenity` absent | `sudo apt install zenity` (repli intégré automatique sinon) |
| Sous-titrage : « aucune parole détectée » | piste instrumentale | relancer en mode `musique` |
| Resolve n'ouvre pas un rush | H.264 non supporté (Linux, version gratuite) | convertir en DNxHR via le module **Conversion** |
| `objet resolve introuvable` | script lancé depuis un terminal externe | utiliser *Workspace → Scripts* ou la console |
| `aucune timeline active` | aucune timeline ouverte | double-cliquer une timeline, relancer |
| Script absent de *Workspace → Scripts* | dossier Scripts lu au démarrage | redémarrer Resolve (une seule fois) |
| Marqueurs invisibles sur la règle | timeline sans focus | cliquer sur la timeline, relancer le script |

Logs utiles :

```bash
tail -f /tmp/edittinghub.log
tail -f ~/.local/share/DaVinciResolve/logs/ResolveDebug.txt
```

---

## Limites connues

- **Resolve Free sous Linux** : l'import FCPXML/EDL ne restitue pas les
  marqueurs ; le scripting externe renvoie `resolve = None`. D'où le passage par
  le menu Scripts / la console.
- **Rang 3** : très dense (une micro-variation tous les 2-3 dixièmes). Décoché
  par défaut pour garder une timeline lisible.
- Le module **Conversion** et le module **Sous-titrage** dépendent de scripts
  externes au dépôt (`ResolveConvert/`, `ResolveSubs/`). Sans eux, seuls les
  modules **Plans** et **Audio** sont opérationnels.
- Pas de tests automatisés dans le dépôt : la validation se fait par les
  scénarios E2E décrits dans `AGENTS.md`.

---

## Licence

Usage personnel. Aucune donnée ne quitte la machine.
