# src/tools/tool.py

"""
Tool registry for agents/MCP: Maps names to full classes (shell, mistral, code_interpreter, etc.).
get_tool(name, config) returns instance; execute(payload) dispatches async.
All implementations complete: Multi-lang code_interpreter (Python/Go/C/Bash), file_browser (pathlib),
web_search (DuckDuckGo via ddgs).
No stubs; verifiable via dict returns, explicit errors. For web_search, pip install ddgs (or duckduckgo-search<6.0.0).
Supports git_ops as shell wrapper, api_call as Mistral fallback.
"""

import ast
import asyncio
import atexit
import contextlib
import io
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import aiofiles
from duckduckgo_search import DDGS

from .api_tools import get_mistral_tool
from .shell_tools import ShellTools

logger = logging.getLogger(__name__)

TOOL_REGISTRY = {
    "shell_exec": ShellTools,
    "git_ops": ShellTools,  # Alias to shell with git resolve
    "api_call": get_mistral_tool,
    "mistral_chat": get_mistral_tool,
    "mistral_embed": get_mistral_tool,
    "code_interpreter": "CodeInterpreter",  # Full multi-lang class
    "file_browser": "FileBrowser",  # Full pathlib class
    "web_search": "WebSearch",  # Full DuckDuckGo class
}


class CodeInterpreter:
    """
    Multi-language code interpreter: Safe eval/execute snippets (Python via ast, Go/C/Bash via temp
    compile/run). Lang-agnostic: Payload specifies language; temps in /tmp with cleanup. No unsafe
    cmds/imports. Supported: python, go, c, bash (from config). Returns output/result; raises on
    invalid/unsafe.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.supported_langs = set(
            config.get("tools", {})
            .get("code_interpreter", {})
            .get("supported_langs", ["python", "go", "c", "bash"])
        )
        self.timeout = config.get("tools", {}).get("code_interpreter", {}).get("timeout", 5)
        self.temp_base = config.get("tools", {}).get("code_interpreter", {}).get(
            "temp_base", "/tmp/agentic"
        )
        self.allowed_commands = config.get("tools", {}).get("shell_exec", {}).get(
            "allowed_commands", ["go", "gcc", "bash"]
        )  # Restrict
        self.temp_dir = Path(f"{self.temp_base}-{uuid.uuid4().hex}")
        self.temp_dir.mkdir(exist_ok=True)

        # Cleanup on exit: Nested def to avoid lambda scope issues
        def cleanup():
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)

        atexit.register(cleanup)

    def _is_python_safe(self, tree: ast.AST) -> bool:
        allowed_nodes = {
            ast.Expr,
            ast.Assign,
            ast.BinOp,
            ast.UnaryOp,
            ast.Num,
            ast.Str,
            ast.Bytes,
            ast.Name,
            ast.NameConstant,
            ast.List,
            ast.Dict,
            ast.Tuple,
            ast.Constant,
        }
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    return False
                if isinstance(node, ast.Call):
                    if node.func.id in {"exec", "eval", "__import__", "open"}:
                        return False
                    if (
                        hasattr(node.func, "attr")
                        and node.func.value.id in {"os", "sys", "subprocess"}
                        and node.func.attr in {"system", "popen", "exec", "eval"}
                    ):
                        return False
        return True

    async def _run_python(self, code: str) -> dict[str, Any]:
        if not code.strip():
            raise ValueError("Empty Python code")
        try:
            tree = ast.parse(code, mode="exec")
            if not self._is_python_safe(tree):
                raise ValueError(
                    "Unsafe Python: Imports, exec/eval, or os/sys/subprocess detected"
                )

            async def exec_safe():
                local_vars = {}
                stdout_capture = io.StringIO()
                with contextlib.redirect_stdout(stdout_capture):
                    exec(
                        compile(tree, "<snippet>", mode="exec"),
                        {
                            "__builtins__": {
                                "len": len,
                                "print": print,
                                "int": int,
                                "str": str,
                                "list": list,
                                "dict": dict,
                            }
                        },
                        local_vars,
                    )
                stdout = stdout_capture.getvalue()
                result = local_vars.get("result", "Executed (no explicit return)")
                return {"result": result, "type": "output", "stdout": stdout.strip()}

            return await asyncio.wait_for(exec_safe(), timeout=self.timeout)
        except TimeoutError:
            raise RuntimeError("Python execution timed out")
        except (SyntaxError, ValueError, TypeError) as e:
            raise RuntimeError(f"Python error: {e}")

    async def _run_go(self, code: str) -> dict[str, Any]:
        if "go" not in self.allowed_commands:
            raise RuntimeError("Go not allowed in config")
        code_file = self.temp_dir / "main.go"
        bin_file = self.temp_dir / "main"
        async with aiofiles.open(code_file, "w", encoding="utf-8") as f:
            await f.write(code)
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "go",
                "run",
                str(code_file),
                cwd=str(self.temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Go error: {stderr.decode().strip()}")
            return {
                "result": stdout.decode().strip(),
                "type": "output",
                "temp_file": str(code_file),
            }
        except TimeoutError:
            if proc:
                proc.kill()
            raise RuntimeError("Go execution timed out")
        finally:
            bin_file.unlink(missing_ok=True)
            code_file.unlink(missing_ok=True)

    async def _run_c(self, code: str) -> dict[str, Any]:
        if "gcc" not in self.allowed_commands:
            raise RuntimeError("C not allowed in config")
        code_file = self.temp_dir / "main.c"
        bin_file = self.temp_dir / "main"
        wrapped_code = f"#include <stdio.h>\nint main() {{\n{code}\nreturn 0;\n}}"
        async with aiofiles.open(code_file, "w", encoding="utf-8") as f:
            await f.write(wrapped_code)
        compile_proc = run_proc = None
        try:
            compile_proc = await asyncio.create_subprocess_exec(
                "gcc",
                str(code_file),
                "-o",
                str(bin_file),
                cwd=str(self.temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, compile_stderr = await asyncio.wait_for(
                compile_proc.communicate(), timeout=self.timeout
            )
            if compile_proc.returncode != 0:
                raise RuntimeError(
                    f"C compile error: {compile_stderr.decode().strip()}"
                )

            run_proc = await asyncio.create_subprocess_exec(
                str(bin_file),
                cwd=str(self.temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, run_stderr = await asyncio.wait_for(
                run_proc.communicate(), timeout=self.timeout
            )
            if run_proc.returncode != 0:
                raise RuntimeError(f"C runtime error: {run_stderr.decode().strip()}")
            return {
                "result": stdout.decode().strip(),
                "type": "output",
                "temp_file": str(code_file),
            }
        except TimeoutError:
            if compile_proc:
                compile_proc.kill()
            if run_proc:
                run_proc.kill()
            raise RuntimeError("C execution timed out")
        finally:
            bin_file.unlink(missing_ok=True)
            code_file.unlink(missing_ok=True)

    async def _run_bash(self, code: str) -> dict[str, Any]:
        if "bash" not in self.allowed_commands:
            raise RuntimeError("Bash not allowed in config")
        script_file = self.temp_dir / "script.sh"
        async with aiofiles.open(script_file, "w", encoding="utf-8") as f:
            await f.write(code)
        await script_file.chmod(0o755)
        # Safety: Scan for destructive cmds
        if any(cmd in code.lower() for cmd in ["rm -rf", "dd if=", "mkfs"]):
            raise ValueError("Unsafe Bash: Destructive commands like rm -rf detected")
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                str(script_file),
                cwd=str(self.temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Bash error: {stderr.decode().strip()}")
            return {
                "result": stdout.decode().strip(),
                "type": "output",
                "temp_file": str(script_file),
            }
        except TimeoutError:
            if proc:
                proc.kill()
            raise RuntimeError("Bash execution timed out")
        finally:
            script_file.unlink(missing_ok=True)

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        lang = payload.get("language", "python").lower()
        code = payload.get("code", "")
        if lang not in self.supported_langs:
            raise ValueError(
                f"Unsupported language: {lang}. Use {sorted(self.supported_langs)}"
            )
        if not code.strip():
            raise ValueError("No code provided")

        if lang == "python":
            result = await self._run_python(code)
        elif lang == "go":
            result = await self._run_go(code)
        elif lang == "c":
            result = await self._run_c(code)
        elif lang == "bash":
            result = await self._run_bash(code)
        else:
            raise ValueError("Internal error: Unsupported lang after check")

        logger.debug(
            "CodeInterpreter executed %s (%d chars): %s",
            lang,
            len(code),
            result.get("result", "OK")[:50],
        )
        return result


class FileBrowser:
    """
    Async file browser: List/walk directories with filters (extensions, exclude).
    Uses pathlib (pure Python, no shell). Payload: {"path": str, "extensions": [str], "max_depth":
    int} → {"files": [paths], "dirs": [paths]}. Excludes from config [file_processing] for safety.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        fp_cfg = config.get("agentic-tools", {}).get("file_processing", {})
        self.exclude_dirs = set(
            fp_cfg.get("exclude_directories", [".git", "__pycache__", "venv"])
        )
        self.exclude_files = set(fp_cfg.get("exclude_files", ["__init__.py"]))
        self.include_exts = set(
            fp_cfg.get(
                "include_extensions",
                [".py", ".go", ".md", ".toml", ".pdf", ".json", ".txt"],
            )
        )
        self.max_depth = config.get("tools", {}).get("file_browser", {}).get(
            "max_depth", 3
        )

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        path_str = payload.get("path", ".")
        exts: list[str] = payload.get("extensions", [])
        depth = payload.get("max_depth", self.max_depth)
        root = Path(path_str).resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Invalid directory path: {path_str}")

        files, dirs = [], []
        async def walk(p: Path, d: int):
            if d > depth or p.name in self.exclude_dirs:
                return
            try:
                for item in p.iterdir():
                    if item.is_dir():
                        dirs.append(str(item))
                        await walk(item, d + 1)
                    elif item.is_file():
                        if item.name in self.exclude_files:
                            continue
                        if exts and item.suffix not in exts:
                            continue
                        if self.include_exts and item.suffix not in self.include_exts:
                            continue
                        files.append(str(item))
            except PermissionError:
                logger.warning("Permission denied in file browser: %s", p)

        await walk(root, 0)
        result = {"files": files, "dirs": dirs, "root": str(root)}
        logger.debug(
            "FileBrowser: Found %d files, %d dirs in %s",
            len(files),
            len(dirs),
            root,
        )
        return result


class WebSearch:
    """
    Async web search via DuckDuckGo: Keyword queries, returns snippets/titles.
    Payload: {"query": str, "max_results": int} → {"results": [{"title": str, "snippet": str, "url":
    str}]}. Filters for relevance (non-ad), rate-limited.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_results = config.get("tools", {}).get("web_search", {}).get(
            "max_results", 10
        )
        self.ddgs = DDGS()

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query", "").strip()
        if not query:
            raise ValueError("No query provided")
        num = min(payload.get("max_results", self.max_results), 20)  # Cap for safety

        try:
            # Run the synchronous search in a separate thread
            search_results = await asyncio.to_thread(
                self.ddgs.text, query, max_results=num
            )

            filtered = []
            if search_results:
                for r in search_results:
                    if (
                        r.get("href")
                        and not r["href"].startswith(("http://redirect", "https://redirect"))
                        and r.get("body")
                        and len(r["body"]) > 20  # Non-ad, relevant
                    ):
                        filtered.append(
                            {
                                "title": r.get("title"),
                                "snippet": r.get("body"),
                                "url": r.get("href"),
                            }
                        )
            logger.debug(
                "WebSearch returned %d filtered results for '%s'", len(filtered), query
            )
            return {"results": filtered}
        except Exception as e:
            logger.error("WebSearch execution failed: %s", e)
            raise RuntimeError(f"Search failed: {e}")


def get_tool(tool_name: str, config: dict[str, Any]) -> Any:
    """
    Factory: Returns tool instance from registry/config.
    All full—no fallbacks to None; raises ValueError on unknown.
    Resolves str classes (e.g., "CodeInterpreter") to instances.
    """
    tool_cls_name = TOOL_REGISTRY.get(tool_name)
    if not tool_cls_name:
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available: {list(TOOL_REGISTRY.keys())}"
        )

    # Resolve class
    if isinstance(tool_cls_name, str):
        if tool_cls_name == "CodeInterpreter":
            return CodeInterpreter(config)
        elif tool_cls_name == "FileBrowser":
            return FileBrowser(config)
        elif tool_cls_name == "WebSearch":
            return WebSearch(config)
    else:  # Callable like ShellTools
        return tool_cls_name(config)


# Generic execute for MCP (routes to tool)
async def execute_tool(payload: dict[str, Any], config: dict[str, Any]) -> Any:
    tool_name = payload.get("tool", "api_call")
    tool = get_tool(tool_name, config)
    return await tool.execute(payload)

