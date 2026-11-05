"""
Global Pytest Configuration and Test Environment Setup.
Configures test-safe environment variables before module imports.
"""

import os

os.environ["API_KEY"] = "test_configured_api_key_2026"

