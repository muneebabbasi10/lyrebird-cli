# 🐦 Lyrebird CLI

An AI-powered coding assistant that brings advanced code generation, debugging, refactoring, and explanation capabilities directly to your terminal. Like Google's Gemini CLI and Anthropic's Claude Code, but open-source and flexible.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## ✨ Features

* **🌟 Code Generation**: Generate code from natural language prompts with context awareness
* **🐛 Bug Fixing**: Automatically detect and fix syntax errors, logical bugs, and code issues
* **♻️ Code Refactoring**: Improve code quality, readability, performance, and maintainability
* **📚 Code Explanation**: Get detailed explanations of code functionality, algorithms, and patterns
* **🏗️ Codebase Analysis**: Summarize project architecture, structure, and design patterns
* **🔌 Multiple Providers**: Support for OpenRouter (GPT-4, Claude, etc.) and DeepSeek APIs
* **🎨 Rich Output**: Beautiful syntax highlighting, progress indicators, and formatted display
* **📁 Flexible Input**: Read from files, stdin, or direct prompts seamlessly
* **⚙️ JSON Output**: Machine-readable output format for automation and scripting
* **🎛️ Explicit Provider Selection**: Choose your preferred AI provider for optimal results

## 🚀 Quick Start

### Installation

```bash
pip install lyrebird-cli
```

Or install from source:

```bash
git clone https://github.com/muneebabbasi10/lyrebird-cli.git
cd lyrebird-cli
pip install -e .
```

### Setup API Keys (Windows PowerShell)

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
$env:DEEPSEEK_API_KEY = "sk-..."
```

> Get your API keys:
>
> * [OpenRouter](https://openrouter.ai/)
> * [DeepSeek](https://platform.deepseek.com/)

### Your First Commands

```bash
lyrebird generate "create a binary search algorithm" --provider openrouter
lyrebird fix broken_script.py --provider deepseek
lyrebird summarize ./my_project --provider openrouter
```

## 📖 Full Usage Guide

All commands require `--provider openrouter` or `--provider deepseek`.

### Code Generation

```bash
lyrebird generate "Create a FastAPI server with user authentication" --provider openrouter
lyrebird generate "Implement quicksort" --lang java --provider deepseek
lyrebird generate "Add logging" --file utils.py --provider openrouter
lyrebird generate "Build a neural network" --provider openrouter --model openai/gpt-4o
```

### Bug Fixing

```bash
lyrebird fix buggy_script.py --provider deepseek
cat broken_code.py | lyrebird fix --provider openrouter
lyrebird fix problematic_code.js --provider deepseek --verbose
lyrebird fix --format json --provider openrouter < code_with_errors.py
```

### Code Refactoring

```bash
lyrebird refactor legacy_code.py --provider openrouter
cat messy_code.py | lyrebird refactor --provider deepseek
lyrebird refactor slow_algorithm.py --provider openrouter --verbose
lyrebird refactor --format json old_code.py --provider deepseek
```

### Code Explanation

```bash
lyrebird explain sorting_algorithm.py --provider deepseek
cat complex_function.py | lyrebird explain --provider openrouter
lyrebird explain --format json mysterious_code.py --provider deepseek
lyrebird explain advanced_topic.py --provider openrouter --verbose
```

### Codebase Analysis

```bash
lyrebird summarize ./my_project --provider openrouter
lyrebird summarize ./backend --provider openrouter --model openai/gpt-4o
lyrebird summarize ./codebase --format json --provider deepseek
lyrebird summarize ./src --provider openrouter --verbose
```

## 🔧 Configuration

### .env File

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
DEEPSEEK_API_KEY=sk-your-deepseek-key
LYREBIRD_DEFAULT_PROVIDER=openrouter
LYREBIRD_DEFAULT_MODEL=openai/gpt-4o
```

### Provider-Specific Models

* **OpenRouter:** `openai/gpt-4o`, `anthropic/claude-3.5-sonnet`, `google/gemini-pro-1.5`, `openai/gpt-3.5-turbo`, `meta-llama/llama-3-70b`
* **DeepSeek:** `deepseek-chat`, `deepseek-coder`

## 📊 Workflows

### Daily Dev

```bash
lyrebird explain src/main.py --provider deepseek
lyrebird generate "Add user profile" --file models/user.py --provider openrouter
lyrebird fix src/profiles.py --provider deepseek
lyrebird refactor src/profiles.py --provider openrouter
lyrebird summarize . --provider openrouter > daily_report.md
```

### Code Review

```bash
lyrebird explain feature_branch.py --provider deepseek --format json
lyrebird fix --format json src/ --provider openrouter | jq '.issues'
lyrebird refactor pull_request.py --provider deepseek --verbose
```

### Learn New Codebase

```bash
lyrebird summarize ./new_project --provider openrouter
lyrebird explain src/core/engine.py --provider deepseek
lyrebird explain design_patterns/ --provider openrouter --verbose
```

### Debugging

```bash
lyrebird explain failing_test.py --provider deepseek
lyrebird generate "Add debugging logs" --file problematic_module.py --provider openrouter
lyrebird fix problematic_module.py --provider deepseek
lyrebird explain fixed_module.py --provider openrouter
```

## 🎨 Output Formats

### Text (Default)

Formatted, colorized CLI output

### JSON

```bash
lyrebird generate "hello world" --format json --provider deepseek
```

Example:

```json
{
  "task": "generate",
  "content": "generated code here",
  "timestamp": "/cwd",
  "version": "0.1.0"
}
```

## 🛠️ Development

### Setup

```bash
git clone https://github.com/muneebabbasi10/lyrebird-cli.git
cd lyrebird-cli
pip install -e ".[dev]"
pre-commit install
```

### Testing

```bash
pytest
pytest --cov=src --cov-report=html
pytest tests/test_cli.py -v
```

### Code Quality

```bash
black src tests
flake8 src tests
mypy src
make quality
```

## 📊 Performance Tips

* **GPT-4o**: best quality, slower
* **Claude 3.5**: great for reasoning
* **GPT-3.5**: fast, cheaper
* **DeepSeek Coder**: high code quality

## ⚠️ Troubleshooting

### API Key Error

```powershell
$env:OPENROUTER_API_KEY = "your-key"
```

### Missing Provider

```bash
# Wrong
lyrebird generate "hello"

# Right
lyrebird generate "hello" --provider openrouter
```

### File Issues

```bash
ls your_file.py
lyrebird fix /full/path/to/file.py --provider deepseek
```

### Rate Limits

```bash
lyrebird generate "code" --provider openrouter --verbose
lyrebird generate "code" --provider deepseek
```

### Debug Logs

```bash
lyrebird generate "test" --provider openrouter --verbose
```

## 📦 Install

```bash
pip install lyrebird-cli
```

## 📄 License

Apache 2.0 - See [LICENSE](https://github.com/apache/httpd/blob/trunk/LICENSE)

## 🤝 Contributing

1. Fork
2. Create branch
3. Add code + tests
4. Ensure tests pass
5. Format with `black`
6. Push & PR

### Guidelines

* PEP 8
* Small focused PRs
* Tests & Docs

## 🙏 Acknowledgments

* Inspired by Gemini CLI, Claude CLI
* Built with Typer, Rich
* Powered by OpenRouter + DeepSeek

## 📒 Related Projects

* [Claude Code CLI](https://www.anthropic.com/claude-code)
* [Gemini CLI](https://ai.google.dev/gemini-api/docs/cli)

## 🛡️ Support

* Bugs: [GitHub Issues](https://github.com/muneebabbasi10/lyrebird-cli/issues)
* Features: [GitHub Discussions](https://github.com/muneebabbasi10/lyrebird-cli/)
* Email: \[abbasimuneeb54@gmail.com]



**Made with ❤️ by [Muneeb](https://github.com/muneebabbasi10). Happy coding! 🚀**
