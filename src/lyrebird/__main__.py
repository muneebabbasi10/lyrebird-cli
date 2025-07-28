#!/usr/bin/env python3
"""
Main entry point for Lyrebird CLI when run as a module.
Allows running with: python -m lyrebird
"""

from .cli import app

if __name__ == "__main__":
    app()