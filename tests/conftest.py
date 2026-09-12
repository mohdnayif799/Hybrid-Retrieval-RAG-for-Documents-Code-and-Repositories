import os
import sys

# Repo root on sys.path so `import src.*` works however pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
