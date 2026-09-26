"""
EasyOCR must not load until a document actually needs OCR.

``import easyocr`` pulls in torchvision, OpenCV, scikit-image and SciPy. Only
scanned PDF pages need any of it, so importing ``src.data_ingestion`` (which
app.py does for every session) must leave it unloaded.

Runs in a fresh interpreter: this test process may already have imported
easyocr through some other path, which would make an in-process check
meaningless.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_importing_data_ingestion_does_not_import_easyocr():
    pytest.importorskip("easyocr")  # needs the full dependency set, as in CI

    result = _run(
        "import sys\n"
        "import src.data_ingestion\n"
        "print('easyocr' in sys.modules)\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"
