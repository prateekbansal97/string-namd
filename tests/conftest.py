# tests/conftest.py

import sys
from pathlib import Path

# Ensure the project root (one level up) is on sys.path so pytest can import string_namd
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
