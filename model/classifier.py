"""
Fake-news classifier entry point.
Delegates to the GNN backend (gnn_classifier.py).
The classify() signature is unchanged so central_receiver.py needs no edits.
"""

from gnn_classifier import load_classifier, classify  # noqa: F401
