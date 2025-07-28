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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)


class Provider(str, Enum):
    openrouter = "openrouter"
    deepseek = "deepseek"


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


class LyrebirdClient:
    """Client for interacting with LLM APIs"""

    def __init__(self, provider: Provider, model: Optional[str] = None, verbose: bool = False):
        self.provider = provider
        self.verbose = verbose

        if verbose:
            logger.setLevel(logging.DEBUG)

        # Set up API client based on provider
        if provider == Provider.openrouter:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                console.print("[red]Error: OPENROUTER_API_KEY environment variable not set[/red]")
                raise typer.Exit(1)

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            self.model = model or "openai/gpt-4"

        elif provider == Provider.deepseek:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                console.print("[red]Error: DEEPSEEK_API_KEY environment variable not set[/red]")
                raise typer.Exit(1)

            self.client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=api_key
            )
            self.model = model or "deepseek-chat"

        logger.debug(f"Initialized {provider} client with model: {self.model}")

    def make_request(self, system_prompt: str, user_prompt: str) -> str:
        """Make API request to LLM"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            logger.debug(f"Making API request with {len(messages)} messages")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1
            )

            result = response.choices[0].message.content
            logger.debug(f"Received response with {len(result)} characters")

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
    rich_markup_mode="rich"
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
            return file.read_text(encoding='utf-8')
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
        return json.dumps({
            "task": task,
            "content": content,
            "timestamp": str(Path().cwd()),
            "version": __version__
        }, indent=2)

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
            help="Show version and exit"
        )
):
    """🐦 Lyrebird CLI: AI-Powered Coding Assistant in Your Terminal"""
    pass


@app.command()
def generate(
        prompt: str = typer.Argument(..., help="Task description for code generation"),
        provider: Provider = typer.Option(
            ...,
            "--provider", "-p",
            help="API provider to use (openrouter or deepseek)"
        ),
        file: Optional[Path] = typer.Option(None, "--file", "-f", help="Input file for context"),
        language: str = typer.Option("python", "--lang", "-l", help="Programming language"),
        model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
        output_format: OutputFormat = typer.Option(OutputFormat.text, "--format", help="Output format"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Generate code from a natural language prompt"""

    client = LyrebirdClient(provider, model, verbose)

    # Read additional context if file provided
    context = ""
    if file:
        context = read_input(file)
        context_info = f"\n\nAdditional context from {file}:\n```{language}\n{context}\n```"
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
            console=console
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
def fix(
        file: Optional[Path] = typer.Argument(None, help="File to fix (or read from stdin)"),
        provider: Provider = typer.Option(
            ...,
            "--provider", "-p",
            help="API provider to use (openrouter or deepseek)"
        ),
        model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
        output_format: OutputFormat = typer.Option(OutputFormat.text, "--format", help="Output format"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
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
        ext_map = {'.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'cpp', '.c': 'c'}
        language = ext_map.get(file.suffix.lower(), 'python')

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
            console=console
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
        file: Optional[Path] = typer.Argument(None, help="File to refactor (or read from stdin)"),
        provider: Provider = typer.Option(
            ...,
            "--provider", "-p",
            help="API provider to use (openrouter or deepseek)"
        ),
        model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
        output_format: OutputFormat = typer.Option(OutputFormat.text, "--format", help="Output format"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
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
        ext_map = {'.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'cpp', '.c': 'c'}
        language = ext_map.get(file.suffix.lower(), 'python')

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
            console=console
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
        file: Optional[Path] = typer.Argument(None, help="File to explain (or read from stdin)"),
        provider: Provider = typer.Option(
            ...,
            "--provider", "-p",
            help="API provider to use (openrouter or deepseek)"
        ),
        model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
        output_format: OutputFormat = typer.Option(OutputFormat.text, "--format", help="Output format"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
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
        ext_map = {'.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'cpp', '.c': 'c'}
        language = ext_map.get(file.suffix.lower(), 'python')

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
            console=console
    ) as progress:
        task = progress.add_task("Analyzing code...", total=None)

        try:
            result = client.make_request(system_prompt, user_prompt)
            progress.remove_task(task)

            if output_format == OutputFormat.text:
                console.print(Panel(result, title="Code Explanation", border_style="blue"))
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
            ...,
            "--provider", "-p",
            help="API provider to use (openrouter or deepseek)"
        ),
        model: Optional[str] = typer.Option(None, "--model", "-m", help="Model to use"),
        output_format: OutputFormat = typer.Option(OutputFormat.text, "--format", help="Output format"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Summarize codebase structure and architecture"""

    client = LyrebirdClient(provider, model, verbose)

    if not directory.exists() or not directory.is_dir():
        console.print(f"[red]Error: {directory} is not a valid directory[/red]")
        raise typer.Exit(1)

    # Collect code files
    code_files = []
    extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.ts', '.jsx', '.tsx'}

    for ext in extensions:
        code_files.extend(directory.rglob(f'*{ext}'))

    if not code_files:
        console.print(f"[yellow]No code files found in {directory}[/yellow]")
        raise typer.Exit(0)

    # Limit files to avoid token limits
    code_files = code_files[:20]  # Limit to first 20 files

    file_info = []
    total_size = 0

    for file_path in code_files:
        try:
            content = file_path.read_text(encoding='utf-8')
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
            console=console
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)

        try:
            result = client.make_request(system_prompt, user_prompt)
            progress.remove_task(task)

            if output_format == OutputFormat.text:
                console.print(Panel(result, title=f"Codebase Summary: {directory.name}", border_style="cyan"))
            else:
                formatted_output = format_output(result, output_format, "summarize")
                console.print(formatted_output)

        except Exception as e:
            progress.remove_task(task)
            console.print(f"[red]Analysis failed: {e}[/red]")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()