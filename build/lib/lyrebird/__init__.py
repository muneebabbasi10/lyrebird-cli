"""
Lyrebird CLI: An AI-Powered Coding Assistant in Your Terminal

A modern CLI tool that brings LLM-powered code generation, debugging,
refactoring, and explanation capabilities directly to your terminal.

Supports OpenRouter and DeepSeek APIs for flexible model selection.
"""

__version__ = "0.1.0"
__author__ = "Muneeb Mahfooz Abbasi"
__email__ = "abbasimuneeb54@gmail.com"
__license__ = "Apache-2.0"

from .cli import app, LyrebirdClient, Provider, OutputFormat

__all__ = [
    "app",
    "LyrebirdClient",
    "Provider",
    "OutputFormat",
]
