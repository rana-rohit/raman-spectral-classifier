"""
conftest.py
Ensures the project root is on sys.path for all pytest runs,
regardless of which directory pytest is launched from.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))