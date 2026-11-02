"""Exceptions partagées entre services et serveur."""


class JobCancelled(Exception):
    """Levée quand l'utilisateur annule un job en cours d'exécution."""
