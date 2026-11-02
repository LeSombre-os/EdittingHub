"""Gestion des sous-processus : annulation réelle (SIGTERM → SIGKILL).

Pourquoi ce module : un simple `proc.terminate()` ne suffit pas. Les scripts
`convert.py` et `transcrire.py` lancent eux-mêmes des `ffmpeg`/`ffprobe` en
parallèle ; si l'on tue seulement le processus père, les enfants continuent de
tourner au fond (« arrêt » qui n'arrête rien).

Solution : chaque job lance son processus dans une *nouvelle session*
(`start_new_session=True`). On peut alors tuer tout le groupe avec
`os.killpg(pgid, SIGTERM)`, attendre un court délai, puis `SIGKILL`.

`iter_lines()` fournit en plus une lecture de stdout non bloquante, ce qui
permet de réagir à l'annulation même quand le sous-processus est silencieux
(cas typique de whisper pendant la transcription).
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time

# Délai laissé au processus pour mourir proprement avant le SIGKILL
GRACE_SECONDS = 3.0


def kill_group(proc: subprocess.Popen | None, grace: float = GRACE_SECONDS) -> bool:
    """Tue le processus **et tous ses enfants** (groupe de processus).

    Retourne True si un signal a été envoyé.
    """
    if proc is None:
        return False
    if proc.poll() is not None:
        # Déjà terminé, mais des petits-enfants peuvent survivre : on tente
        # quand même le kill du groupe (ignoré si le groupe n'existe plus).
        _signal_group(proc, signal.SIGKILL)
        return True

    sent = _signal_group(proc, signal.SIGTERM)
    deadline = time.time() + max(0.5, grace)
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    if proc.poll() is None:
        _signal_group(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    else:
        # Le père est mort : on nettoie les éventuels orphelins du groupe.
        _signal_group(proc, signal.SIGKILL)
    return sent


def _signal_group(proc: subprocess.Popen, sig: int) -> bool:
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = None

    if pgid is not None:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except (PermissionError, OSError):
            pass

    # Repli : pas de groupe accessible → on tue au moins le père
    try:
        proc.send_signal(sig)
        return True
    except (ProcessLookupError, OSError):
        return False


def iter_lines(proc: subprocess.Popen, cancel_event=None, poll: float = 0.2,
               on_cancel=None):
    """Itère sur stdout sans jamais bloquer indéfiniment.

    Args:
        proc : processus lancé avec stdout=PIPE, text=True.
        cancel_event : threading.Event ; si levé, le groupe est tué puis
                       l'itération s'arrête.
        poll : intervalle de vérification de l'annulation (secondes).
        on_cancel : callback appelé avant le kill (logs, nettoyage…).
    """
    q: queue.Queue = queue.Queue()
    done = threading.Event()

    def _reader():
        try:
            for line in proc.stdout:
                q.put(line)
        except (ValueError, OSError):
            # Pipe fermé brutalement (processus tué)
            pass
        finally:
            done.set()
            q.put(None)

    reader = threading.Thread(target=_reader, daemon=True,
                              name=f"pipe-{proc.pid}")
    reader.start()

    while True:
        if cancel_event is not None and cancel_event.is_set():
            if on_cancel:
                on_cancel()
            kill_group(proc)
            return
        try:
            item = q.get(timeout=poll)
        except queue.Empty:
            if done.is_set() and proc.poll() is not None:
                return
            continue
        if item is None:
            return
        yield item

    # (le `return` ci-dessus laisse le thread lecteur se terminer seul)
