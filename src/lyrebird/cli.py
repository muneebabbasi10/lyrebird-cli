#!/usr/bin/env python3
"""
Lyrebird CLI: An AI-Powered Coding Assistant in Your Terminal
Main CLI module using Typer framework
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional, List
from enum import Enum

import typer
import ollama
from openai import OpenAI
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

__version__ = "0.1.0"

# Initialize Rich console for better output formatting
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


class Provider(str, Enum):
    openrouter = "openrouter"
    deepseek = "deepseek"
    ollama = "ollama"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class LyrebirdClient:
    """
    Client for interacting with Large Language Model (LLM) APIs.

    This class provides a unified interface to interact with different LLM providers
    including OpenRouter, DeepSeek, and Ollama. It handles authentication, model selection,
    and request formatting.

    Attributes:
        provider (Provider): The LLM service provider to use
        verbose (bool): Flag to enable verbose logging
        model (str): The specific model to use for requests
        client (Union[OpenAI, ollama]): The client instance for the selected provider

    Example:
        >>> client = LyrebirdClient(Provider.ollama, model="codellama", verbose=True)
        >>> response = client.make_request("System prompt", "User query")
    """

    def __init__(
        self, provider: Provider, model: Optional[str] = None, verbose: bool = False
    ):
        """
        Initializes the Lyrebird client for a specific LLM provider.

        Sets up the API client based on the selected provider, handles authentication
        using environment variables, and configures default models. Also sets up
        logging verbosity based on the verbose flag.

        Args:
            provider: The LLM service provider to use (openrouter, deepseek, or ollama)
            model: Optional specific model name. Defaults to provider's recommended model:
                - openrouter: "openai/gpt-4"
                - deepseek: "deepseek-chat"
                - ollama: "codellama"
            verbose: Enable verbose logging if True (default: False)

        Raises:
            typer.Exit(1): If required API key environment variables are not set for:
                - OpenRouter: OPENROUTER_API_KEY
                - DeepSeek: DEEPSEEK_API_KEY

        Environment Variables:
            OPENROUTER_API_KEY: API key for OpenRouter service
            DEEPSEEK_API_KEY: API key for DeepSeek service

        Note:
            Ollama doesn't require API keys as it runs locally

        Example:
            >>> client = LyrebirdClient(Provider.openrouter, verbose=True)
        """
        self.provider = provider
        self.verbose = verbose
        self.model = model

        if verbose:
            logger.setLevel(logging.DEBUG)

        if provider == Provider.openrouter:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                console.print(
                    "[red]Error: OPENROUTER_API_KEY environment variable not set[/red]"
                )
                raise typer.Exit(1)

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=api_key
            )
            self.model = model or "openai/gpt-4"

        elif provider == Provider.deepseek:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                console.print(
                    "[red]Error: DEEPSEEK_API_KEY environment variable not set[/red]"
                )
                raise typer.Exit(1)

            self.client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
            self.model = model or "deepseek-chat"

        elif provider == Provider.ollama:
            self.client = ollama
            self.model = model or "codellama"

            # Check and pull Ollama model if needed
            self._ensure_ollama_model_exists()

        logger.debug(f"Initialized {provider} client with model: {self.model}")

    def _ensure_ollama_model_exists(self):
        """
        Ensures the requested Ollama model exists locally, pulling it if necessary.

        This method performs the following steps:
        1. Retrieves the list of locally available Ollama models
        2. Checks if the requested model is in the local model list
        3. If missing, initiates a pull operation with progress tracking
        4. Handles pull errors and provides user feedback

        The progress tracking includes:
        - Download percentage visualization
        - Verification status updates
        - Completion notifications

        Raises:
            typer.Exit(1): If model verification or pulling fails

        Note:
            This method is automatically called during Ollama client initialization
            and should not be called directly by users.

        Example Workflow:
            When initializing with a new model:
            >>> client = LyrebirdClient(Provider.ollama, model="llama3")
            # If 'llama3' is missing:
            [yellow]Model 'llama3' not found locally. Pulling from Ollama...[/yellow]
            Pulling llama3: downloading ███████████████████ 100%
            [bold green]✓ Successfully pulled llama3[/bold green]
        """
        try:
            # Check if model exists locally
            local_models = ollama.list().get("models", [])
            model_exists = any(m["model"] == self.model for m in local_models)

            if not model_exists:
                console.print(
                    f"[yellow]Model '{self.model}' not found locally. Pulling from Ollama...[/yellow]"
                )

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Pulling {self.model}...", total=None)

                    # Stream the pull progress
                    for progress_info in ollama.pull(self.model, stream=True):
                        if "status" in progress_info:
                            status = progress_info["status"]
                            progress.update(
                                task, description=f"Pulling {self.model}: {status}"
                            )

                    progress.update(
                        task, description=f"[green]Model '{self.model}' ready![/green]"
                    )
                    progress.stop()

                console.print(
                    f"[bold green]✓ Successfully pulled {self.model}[/bold green]"
                )
            else:
                logger.debug(f"Model '{self.model}' found locally")

        except Exception as e:
            console.print(f"[red]Error verifying/pulling Ollama model: {e}[/red]")
            raise typer.Exit(1)

    def make_request(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends a chat completion request to the selected LLM provider.

        Args:
            system_prompt (str): The system role content.
            user_prompt (str): The user's input prompt.

        Returns:
            str: The response content from the model.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            logger.debug(f"Making API request with provider: {self.provider}")

            if self.provider in [Provider.openrouter, Provider.deepseek]:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.1
                )
                result = response.choices[0].message.content

            elif self.provider == Provider.ollama:
                response = self.client.chat(model=self.model, messages=messages)
                result = response["message"]["content"]

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            logger.debug(f"Received response: {result[:80]}...")  # log first 80 chars
            return result

        except Exception as e:
            logger.error(f"API request failed: {e}")
            console.print(f"[red]API Error: {e}[/red]")
            raise typer.Exit(1)


# Initialize Typer app
app = typer.Typer(
    name="lyrebird",
    help="🐦 Lyrebird CLI: AI-Powered Coding Assistant in Your Terminal",
    add_completion=False,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    """Show version information"""
    if value:
        console.print(f"Lyrebird CLI v{__version__}")
        raise typer.Exit()


def read_input(file: Optional[Path] = None) -> str:
    """Read input from file or stdin"""
    if file:
        if not file.exists():
            console.print(f"[red]Error: File {file} does not exist[/red]")
            raise typer.Exit(1)
        try:
            return file.read_text(encoding="utf-8")
        except Exception as e:
            console.print(f"[red]Error reading file {file}: {e}[/red]")
            raise typer.Exit(1)

    # Check if there's input from stdin
    if not sys.stdin.isatty():
        return sys.stdin.read()

    return ""


def format_output(content: str, output_format: OutputFormat, task: str) -> str:
    """Format output according to specified format"""
    if output_format == OutputFormat.json:
        return json.dumps(
            {
                "task": task,
                "content": content,
                "timestamp": str(Path().cwd()),
                "version": __version__,
            },
            indent=2,
        )

    return content


def display_code(content: str, language: str = "python"):
    """Display code with syntax highlighting"""
    syntax = Syntax(content, language, theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Generated Code", border_style="green"))


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """🐦 Lyrebird CLI: AI-Powered Coding Assistant in Your Terminal"""
    pass


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Task description for code generation"),
    provider: Provider = typer.Option(
        ..., "--provider", "-p", help="API provider to use (openrouter or deepseek or ollama)"
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Input file for context"
    ),
    language: str = typer.Option("python", "--lang", "-l", help="Programming language"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--format", help="Output format"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Generate code from a natural language prompt"""

    client = LyrebirdClient(provider, model, verbose)

    # Read additional context if file provided
    context = ""
    if file:
        context = read_input(file)
        context_info = (
            f"\n\nAdditional context from {file}:\n```{language}\n{context}\n```"
        )
    else:
        context_info = ""

    system_prompt = f"""You are an expert coding assistant specializing in {language}. 
Generate clean, well-commented, production-ready code that follows best practices.
Always include proper error handling where appropriate."""

    user_prompt = f"""Generate {language} code for: {prompt}{context_info}

Requirements:
- Write clean, readable code
- Include appropriate comments
- Follow {language} best practices
- Include error handling where needed
- Make the code production-ready"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating code...", total=None)

        try:
            result = client.make_request(system_prompt, user_prompt)
            progress.remove_task(task)

            if output_format == OutputFormat.text:
                display_code(result, language)
            else:
                formatted_output = format_output(result, output_format, "generate")
                console.print(formatted_output)

        except Exception as e:
            progress.remove_task(task)
            console.print(f"[red]Generation failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    provider: Provider = typer.Option(
        ..., "--provider", "-p", help="API provider to use (openrouter or deepseek or ollama)"
    ),
):
    """Start a coding-focused chat session with the assistant."""
    console = Console()
    console.print("[bold green]Starting Chat Session. Type 'exit' to quit.[/]")
    history = []
    # Initialize LyrebirdClient for chat
    try:
        client = LyrebirdClient(Provider(provider), model=model)
    except Exception as e:
        console.print(f"[red]Failed to initialize client: {e}[/red]")
        return

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        # Use LyrebirdClient to generate a completion
        system_prompt = (
            "You are Lyrebird, an expert coding assistant designed for CLI use. "
            "Always provide clear, accurate, and concise responses tailored to software developers. "
            "Prefer code snippets over long explanations. Use bullet points for steps. "
            "If unsure, say so rather than guessing. "
            "When showing code, explain briefly what it does. Avoid unnecessary greetings or sign-offs."
        )
        response = client.make_request(system_prompt, user_input)
        history.append((user_input, response))
        console.print(f"[cyan]Assistant:[/] {response}")


@app.command()
def fix(
    file: Optional[Path] = typer.Argument(
        None, help="File to fix (or read from stdin)"
    ),
    provider: Provider = typer.Option(
        ..., "--provider", "-p", help="API provider to use (openrouter or deepseek)"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--format", help="Output format"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Fix bugs and errors in code"""

    client = LyrebirdClient(provider, model, verbose)

    # Read code to fix
    code = read_input(file)
    if not code.strip():
        console.print("[red]Error: No code provided to fix[/red]")
        raise typer.Exit(1)

    # Detect language from file extension or content
    language = "python"  # default
    if file and file.suffix:
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
        }
        language = ext_map.get(file.suffix.lower(), "python")

    system_prompt = f"""You are an expert {language} developer and debugger.
Fix bugs, syntax errors, and logical issues in the provided code.
Maintain the original functionality while improving code quality."""

    user_prompt = f"""Please fix all bugs and issues in this {language} code:

```{language}
{code}
```

Requirements:
- Fix all syntax errors
- Resolve logical bugs
- Improve error handling
- Maintain original functionality
- Add comments explaining fixes
- Return only the corrected code"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fixing code...", total=None)

        try:
            result = client.make_request(system_prompt, user_prompt)
            progress.remove_task(task)

            if output_format == OutputFormat.text:
                display_code(result, language)
            else:
                formatted_output = format_output(result, output_format, "fix")
                console.print(formatted_output)

        except Exception as e:
            progress.remove_task(task)
            console.print(f"[red]Fix failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def refactor(
    file: Optional[Path] = typer.Argument(
        None, help="File to refactor (or read from stdin)"
    ),
    provider: Provider = typer.Option(
        ..., "--provider", "-p", help="API provider to use (openrouter or deepseek)"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--format", help="Output format"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Refactor code for better performance and readability"""

    client = LyrebirdClient(provider, model, verbose)

    # Read code to refactor
    code = read_input(file)
    if not code.strip():
        console.print("[red]Error: No code provided to refactor[/red]")
        raise typer.Exit(1)

    # Detect language from file extension
    language = "python"  # default
    if file and file.suffix:
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
        }
        language = ext_map.get(file.suffix.lower(), "python")

    system_prompt = f"""You are an expert {language} developer focused on code optimization and best practices.
Refactor code to improve readability, performance, and maintainability while preserving functionality."""

    user_prompt = f"""Please refactor this {language} code for better quality:

```{language}
{code}
```

Refactoring goals:
- Improve code readability
- Optimize performance where possible
- Follow {language} best practices
- Reduce code complexity
- Improve naming conventions
- Add proper documentation
- Maintain all original functionality

Return only the refactored code"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Refactoring code...", total=None)

        try:
            result = client.make_request(system_prompt, user_prompt)
            progress.remove_task(task)

            if output_format == OutputFormat.text:
                display_code(result, language)
            else:
                formatted_output = format_output(result, output_format, "refactor")
                console.print(formatted_output)

        except Exception as e:
            progress.remove_task(task)
            console.print(f"[red]Refactoring failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def explain(
    file: Optional[Path] = typer.Argument(
        None, help="File to explain (or read from stdin)"
    ),
    provider: Provider = typer.Option(
        ..., "--provider", "-p", help="API provider to use (openrouter or deepseek)"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--format", help="Output format"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Explain code functionality and structure"""

    client = LyrebirdClient(provider, model, verbose)

    # Read code to explain
    code = read_input(file)
    if not code.strip():
        console.print("[red]Error: No code provided to explain[/red]")
        raise typer.Exit(1)

    # Detect language from file extension
    language = "python"  # default
    if file and file.suffix:
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
        }
        language = ext_map.get(file.suffix.lower(), "python")

    system_prompt = f"""You are an expert {language} developer and technical communicator.
Provide clear, comprehensive explanations of code functionality, structure, and implementation details."""

    user_prompt = f"""Please explain this {language} code in detail:

```{language}
{code}
```

Explanation should include:
- Overall purpose and functionality
- Key algorithms and data structures used
- Flow of execution
- Important design patterns
- Potential improvements or considerations
- Any notable implementation details
- How different parts work together"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing code...", total=None)

        try:
            result = client.make_request(system_prompt, user_prompt)
            progress.remove_task(task)

            if output_format == OutputFormat.text:
                console.print(
                    Panel(result, title="Code Explanation", border_style="blue")
                )
            else:
                formatted_output = format_output(result, output_format, "explain")
                console.print(formatted_output)

        except Exception as e:
            progress.remove_task(task)
            console.print(f"[red]Explanation failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def summarize(
    directory: Path = typer.Argument(..., help="Directory to summarize"),
    provider: Provider = typer.Option(
        ..., "--provider", "-p", help="API provider to use (openrouter or deepseek)"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
    output_format: OutputFormat = typer.Option(
        OutputFormat.text, "--format", help="Output format"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Summarize codebase structure and architecture"""

    client = LyrebirdClient(provider, model, verbose)

    if not directory.exists() or not directory.is_dir():
        console.print(f"[red]Error: {directory} is not a valid directory[/red]")
        raise typer.Exit(1)

    # Collect code files
    code_files = []
    extensions = {".py", ".js", ".java", ".cpp", ".c", ".h", ".ts", ".jsx", ".tsx"}

    for ext in extensions:
        code_files.extend(directory.rglob(f"*{ext}"))

    if not code_files:
        console.print(f"[yellow]No code files found in {directory}[/yellow]")
        raise typer.Exit(0)

    # Limit files to avoid token limits
    code_files = code_files[:20]  # Limit to first 20 files

    file_info = []
    total_size = 0

    for file_path in code_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            # Truncate very large files
            if len(content) > 2000:
                content = content[:2000] + "\n... (truncated)"
            file_info.append(f"\n--- {file_path.relative_to(directory)} ---\n{content}")
            total_size += len(content)

            # Prevent extremely large prompts
            if total_size > 50000:
                break

        except Exception as e:
            logger.debug(f"Skipping {file_path}: {e}")
            continue

    codebase_content = "\n".join(file_info)

    system_prompt = """You are an expert software architect and code analyst.
Analyze codebases and provide comprehensive architectural summaries."""

    user_prompt = f"""Analyze this codebase and provide a comprehensive summary:

{codebase_content}

Please provide:
- Overall architecture and structure
- Main components and modules
- Technology stack used
- Design patterns observed
- Key functionality and features
- Data flow and interactions
- Code quality observations
- Potential areas for improvement"""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)

        try:
            result = client.make_request(system_prompt, user_prompt)
            progress.remove_task(task)

            if output_format == OutputFormat.text:
                console.print(
                    Panel(
                        result,
                        title=f"Codebase Summary: {directory.name}",
                        border_style="cyan",
                    )
                )
            else:
                formatted_output = format_output(result, output_format, "summarize")
                console.print(formatted_output)

        except Exception as e:
            progress.remove_task(task)
            console.print(f"[red]Analysis failed: {e}[/red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
