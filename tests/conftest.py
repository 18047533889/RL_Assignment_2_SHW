"""
Global pytest path setup: prepend ``assignment2/src`` to ``sys.path``.

Tests can ``import env``, ``import board``, etc., matching ``PYTHONPATH=src`` for training.
Some modules also insert ``src`` locally so files stay runnable in isolation.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
