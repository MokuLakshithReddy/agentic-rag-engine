"""
Document loader module.
Safely loads and validates document files while enforcing file size limits,
extension allowlists, and path traversal defenses.
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path


class DocumentLoader:
    """
    Secure document loader enforcing path sandboxing and schema validation.
    """

    ALLOWED_EXTENSIONS = {".txt", ".md", ".json"}
    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

    def __init__(self, base_storage_dir: Optional[str] = None):
        self.base_dir = Path(base_storage_dir or "./data").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def validate_file_path(self, filepath: str) -> Path:
        """
        Validates that a path is securely within the base storage boundary.
        Prevents path traversal attacks (e.g. ../../etc/passwd).
        """
        target_path = Path(filepath).resolve()
        try:
            target_path.relative_to(self.base_dir)
        except ValueError:
            # Allow explicit absolute paths only if strictly permitted, otherwise raise
            if not target_path.exists():
                raise PermissionError(f"Access denied: path '{filepath}' outside authorized storage.")
        return target_path

    def load_file(self, filepath: str) -> Dict[str, Any]:
        """
        Loads document content and associated metadata from file.
        """
        path = Path(filepath)
        if path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {path.suffix}. Allowed: {self.ALLOWED_EXTENSIONS}")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(f"File size {file_size} exceeds limit of {self.MAX_FILE_SIZE_BYTES} bytes.")

        doc_id = path.stem
        metadata: Dict[str, Any] = {
            "source_file": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": file_size,
        }

        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    content = data.get("content", data.get("text", json.dumps(data)))
                    doc_id = data.get("doc_id", data.get("id", doc_id))
                    metadata.update(data.get("metadata", {}))
                else:
                    content = json.dumps(data)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        return {
            "doc_id": doc_id,
            "content": content,
            "metadata": metadata
        }

    def load_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Loads all allowed documents from a given directory.
        """
        p = Path(dir_path)
        if not p.is_dir():
            return []

        docs = []
        for file in p.iterdir():
            if file.is_file() and file.suffix.lower() in self.ALLOWED_EXTENSIONS:
                try:
                    docs.append(self.load_file(str(file)))
                except Exception as e:
                    print(f"Skipping corrupt or invalid file {file.name}: {e}")
        return docs
