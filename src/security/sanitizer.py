"""
Security Sanitizer Module.
Guards against direct prompt injection, malicious query characters,
and path traversal attacks.
"""

import re
from pathlib import Path


class SecuritySanitizer:
    """
    Sanitizes user input queries and file parameters.
    """

    DANGEROUS_QUERY_PATTERNS = [
        r"ignore (?:all )?previous instructions",
        r"system override",
        r"disregard (?:the )?rules",
        r"you are now (?:dan|an uncensored ai)",
        r"<script.*?>.*?</script>",
        r"eval\s*\(",
        r"__import__",
        r"os\.system"
    ]

    def sanitize_query(self, query: str) -> str:
        """
        Cleans user input, strips control characters and neutralizes injection prompts.
        """
        if not query:
            return ""

        # Remove null bytes and non-printable control characters
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", query)

        # Check for aggressive prompt injection sequences and neutralize
        for pattern in self.DANGEROUS_QUERY_PATTERNS:
            cleaned = re.sub(pattern, "[FILTERED_INJECTION_ATTEMPT]", cleaned, flags=re.IGNORECASE)

        # Limit maximum character length to prevent resource exhaustion DoS
        if len(cleaned) > 2000:
            cleaned = cleaned[:2000]

        return cleaned.strip()

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitizes filenames to prevent path traversal (LFI/RFI).
        """
        # Strip directory separators and relative traversal indicators
        clean_name = Path(filename).name
        clean_name = re.sub(r"\.\.+", ".", clean_name)
        clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean_name)
        return clean_name

