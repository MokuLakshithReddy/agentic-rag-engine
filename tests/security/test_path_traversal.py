"""
Security tests for path traversal prevention during file loading.
"""

import pytest
from src.ingestion.loader import DocumentLoader


def test_path_traversal_detection():
    loader = DocumentLoader()
    with pytest.raises(Exception):
        loader.load_file("../../../../windows/system32/cmd.exe")

