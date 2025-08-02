"""
Unit tests for Lyrebird CLI
"""

import os
import json
import pytest
import typer
from unittest.mock import Mock, patch
from pathlib import Path
from typer.testing import CliRunner

from src.lyrebird.cli import app, LyrebirdClient, Provider, OutputFormat

runner = CliRunner()


class TestLyrebirdClient:
    """Test LyrebirdClient functionality"""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.OpenAI")
    def test_openrouter_client_init(self, mock_openai):
        """Test OpenRouter client initialization"""
        client = LyrebirdClient(Provider.openrouter, verbose=False)
        assert client.provider == Provider.openrouter
        assert client.model == "openai/gpt-4"
        mock_openai.assert_called_once_with(
            base_url="https://openrouter.ai/api/v1", api_key="test-key"
        )

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.OpenAI")
    def test_deepseek_client_init(self, mock_openai):
        """Test DeepSeek client initialization"""
        client = LyrebirdClient(Provider.deepseek, verbose=False)
        assert client.provider == Provider.deepseek
        assert client.model == "deepseek-chat"
        mock_openai.assert_called_once_with(
            base_url="https://api.deepseek.com", api_key="test-key"
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        """Test error handling for missing API key"""
        with pytest.raises(typer.Exit) as exc_info:
            LyrebirdClient(Provider.openrouter)
        assert exc_info.value.exit_code == 1

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.OpenAI")
    def test_make_request(self, mock_openai):
        """Test API request functionality"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated code here"

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = LyrebirdClient(Provider.openrouter, verbose=False)
        result = client.make_request("system prompt", "user prompt")

        assert result == "Generated code here"
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.lyrebird.cli.ollama")
    def test_ollama_client_init(self, mock_ollama):
        """Test Ollama client initialization"""
        # Mock model list response
        mock_ollama.list.return_value = {
            "models": [
                {"model": "codellama", "size": 1000},
                {"model": "llama2", "size": 800},
            ]
        }

        client = LyrebirdClient(Provider.ollama, verbose=False)
        assert client.provider == Provider.ollama
        assert client.model == "codellama"
        mock_ollama.list.assert_called_once()
        mock_ollama.pull.assert_not_called()

    @patch("src.lyrebird.cli.ollama")
    def test_ollama_client_init_missing_model(self, mock_ollama):
        """Test Ollama client pulls missing model"""
        # Mock model list response (missing requested model)
        mock_ollama.list.return_value = {
            "models": [
                {"model": "codellama", "size": 1000},
                {"model": "mistral", "size": 700},
            ]
        }

        # Mock pull progress stream
        mock_pull_stream = [
            {"status": "starting"},
            {"status": "downloading", "completed": 50, "total": 100},
            {"status": "verifying"},
            {"status": "complete"},
        ]
        mock_ollama.pull.return_value = mock_pull_stream

        client = LyrebirdClient(Provider.ollama, model="llama3", verbose=False)
        assert client.model == "llama3"
        mock_ollama.list.assert_called_once()
        mock_ollama.pull.assert_called_once_with("llama3", stream=True)

    @patch("src.lyrebird.cli.ollama")
    def test_ollama_client_pull_error(self, mock_ollama):
        """Test Ollama client handles pull errors"""
        mock_ollama.list.return_value = {"models": []}
        mock_ollama.pull.side_effect = Exception("Network error")

        with pytest.raises(typer.Exit) as exc_info:
            LyrebirdClient(Provider.ollama, model="llama3")

        assert exc_info.value.exit_code == 1
        mock_ollama.pull.assert_called_once()

    @patch("src.lyrebird.cli.ollama")
    def test_ensure_ollama_model_exists_with_progress(self, mock_ollama):
        """Test model pulling shows progress updates"""
        # Mock model list response (missing model)
        mock_ollama.list.return_value = {"models": []}

        # Mock pull progress stream
        mock_pull_stream = [
            {"status": "starting"},
            {"status": "downloading", "completed": 25, "total": 100},
            {"status": "downloading", "completed": 50, "total": 100},
            {"status": "downloading", "completed": 100, "total": 100},
            {"status": "verifying"},
            {"status": "complete"},
        ]
        mock_ollama.pull.return_value = mock_pull_stream

        # Mock console to capture output
        with patch("src.lyrebird.cli.console") as mock_console:
            client = LyrebirdClient(Provider.ollama, model="llama3")

            # Verify progress updates were called
            assert mock_console.print.call_count > 0
            progress_calls = [call[0][0] for call in mock_console.print.call_args_list]
            assert any("Pulling llama3" in str(call) for call in progress_calls)

    @patch("src.lyrebird.cli.ollama")
    def test_ollama_model_override(self, mock_ollama):
        """Test custom model override for Ollama"""
        mock_ollama.list.return_value = {"models": []}

        client = LyrebirdClient(Provider.ollama, model="custom-model")
        assert client.model == "custom-model"

    @patch("src.lyrebird.cli.ollama")
    def test_ollama_default_model(self, mock_ollama):
        """Test default model selection for Ollama"""
        mock_ollama.list.return_value = {"models": [{"model": "codellama"}]}

        client = LyrebirdClient(Provider.ollama)
        assert client.model == "codellama"


class TestCLICommands:
    """Test CLI command functionality"""

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.LyrebirdClient")
    def test_generate_command(self, mock_client_class):
        """Test generate command"""
        mock_client = Mock()
        mock_client.make_request.return_value = "def hello(): print('Hello, World!')"
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            ["generate", "Create a hello world function", "--provider", "openrouter"],
        )

        assert result.exit_code == 0
        mock_client.make_request.assert_called_once()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.LyrebirdClient")
    def test_refactor_command(self, mock_client_class):
        """Test refactor command"""
        mock_client = Mock()
        mock_client.make_request.return_value = (
            "def refactored_function(): return 'refactored'"
        )
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            ["refactor", "--provider", "openrouter"],
            input="def old_function(): return 'old'",
        )

        assert result.exit_code == 0
        mock_client.make_request.assert_called_once()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.LyrebirdClient")
    def test_explain_command(self, mock_client_class):
        """Test explain command"""
        mock_client = Mock()
        mock_client.make_request.return_value = "This function prints hello world"
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            ["explain", "--provider", "openrouter"],
            input="def hello(): print('Hello, World!')",
        )

        assert result.exit_code == 0
        mock_client.make_request.assert_called_once()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.LyrebirdClient")
    def test_summarize_command(self, mock_client_class, tmp_path):
        """Test summarize command using a real temporary directory"""
        mock_client = Mock()
        mock_client.make_request.return_value = (
            "This is a Python project with basic structure"
        )
        mock_client_class.return_value = mock_client

        # Create a real temporary directory structure
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create a file in the directory
        test_file = project_dir / "test.py"
        test_file.write_text("print('test')")

        # Create a subdirectory and file
        sub_dir = project_dir / "subdir"
        sub_dir.mkdir()
        sub_file = sub_dir / "utils.py"
        sub_file.write_text("def helper(): pass")

        result = runner.invoke(
            app, ["summarize", str(project_dir), "--provider", "openrouter"]
        )

        assert result.exit_code == 0

    def test_json_output_format(self):
        """Test JSON output formatting"""
        from src.lyrebird.cli import format_output

        content = "test content"
        result = format_output(content, OutputFormat.json, "test_task")
        parsed = json.loads(result)

        assert parsed["task"] == "test_task"
        assert parsed["content"] == "test content"
        assert "timestamp" in parsed
        assert "version" in parsed

    def test_version_command(self):
        """Test version command"""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Lyrebird CLI v0.1.0" in result.stdout

    @patch("src.lyrebird.cli.sys.stdin")
    def test_read_input_from_stdin(self, mock_stdin):
        """Test reading input from stdin"""
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "test input"

        from src.lyrebird.cli import read_input

        result = read_input()
        assert result == "test input"

    def test_read_input_from_file(self):
        """Test reading input from file"""
        from src.lyrebird.cli import read_input

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("test file content")
            temp_path = Path(f.name)

        try:
            result = read_input(temp_path)
            assert result == "test file content"
        finally:
            temp_path.unlink()

    def test_read_input_nonexistent_file(self):
        """Test error handling for nonexistent file"""
        from src.lyrebird.cli import read_input

        with pytest.raises(typer.Exit) as exc_info:
            read_input(Path("nonexistent_file.py"))
        assert exc_info.value.exit_code == 1

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.LyrebirdClient")
    def test_fix_command_with_stdin(self, mock_client_class):
        """Test fix command with stdin input"""
        mock_client = Mock()
        mock_client.make_request.return_value = "def fixed_function(): pass"
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            app,
            ["fix", "--provider", "openrouter"],
            input="def broken_function(\n    pass",
        )

        assert result.exit_code == 0
        mock_client.make_request.assert_called_once()


class TestErrorHandling:
    """Test error handling scenarios"""

    def test_no_api_key_error(self):
        """Test behavior when no API key is set"""
        with patch.dict(os.environ, {}, clear=True):
            result = runner.invoke(app, ["generate", "test prompt"])
            assert result.exit_code == 1

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("src.lyrebird.cli.LyrebirdClient")
    def test_api_request_failure(self, mock_client_class):
        """Test handling of API request failures"""
        mock_client = Mock()
        mock_client.make_request.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        result = runner.invoke(app, ["generate", "test prompt"])
        assert result.exit_code == 1

    def test_empty_input_for_fix(self):
        """Test fix command with empty input"""
        result = runner.invoke(app, ["fix"], input="")
        assert result.exit_code == 1
