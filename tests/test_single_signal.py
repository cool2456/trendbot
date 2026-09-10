from __future__ import annotations

import ast
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

import trendbot.signal as signal_module
from trendbot.engine import backtest as backtest_module

REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_PATH = REPO_ROOT / "trendbot" / "signal.py"
BACKTEST_PATH = REPO_ROOT / "trendbot" / "engine" / "backtest.py"
RUNNER_PATH = REPO_ROOT / "trendbot" / "runner.py"

SIGNAL_SHA256 = "514570198370cfa3503a3406d62dc552bd99c9ae78a30a3c97e488899af3db60"

_HASH_FAILURE_NOTE = (
    "trendbot/signal.py no longer matches the hash recorded in this test.\n"
    "\n"
    "Per PREREGISTRATION.md, the signal definition is frozen: 'Any change to any\n"
    "number below produces a new strategy with a new version number and a new\n"
    "date, and its results may not be compared to, or substituted for, this one.'\n"
    "\n"
    "Do NOT simply paste the new digest in to make this pass. Either\n"
    "  (a) the signal definition changed -> this is a NEW STRATEGY: bump the\n"
    "      version and date in PREREGISTRATION.md, re-run the section 7 protocol\n"
    "      from step 1, and update SIGNAL_SHA256 as part of that change; or\n"
    "  (b) the change was genuinely cosmetic -> update SIGNAL_SHA256 deliberately,\n"
    "      in its own commit, with the diff visible in review."
)

_SKIP_DIRS = frozenset(
    {
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".eggs",
        "build",
        "dist",
        "node_modules",
        "data",
        "tests",
    }
)


def _repo_python_files() -> list[Path]:
    files = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        files.append(path)
    return files


def _parse(path: Path) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    try:
        return ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"{path.relative_to(REPO_ROOT)} does not parse: {exc}")


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _module_path(modname: str) -> Path | None:
    base = REPO_ROOT.joinpath(*modname.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _direct_trendbot_imports(modname: str) -> set[str]:
    path = _module_path(modname)
    if path is None:
        return set()
    package = modname if path.name == "__init__.py" else modname.rpartition(".")[0]

    found: set[str] = set()
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "trendbot":
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                prefix = ".".join(parts)
                target = f"{prefix}.{node.module}" if node.module else prefix
            else:
                target = node.module or ""
            if not target.startswith("trendbot"):
                continue
            found.add(target)
            for alias in node.names:
                if _module_path(f"{target}.{alias.name}") is not None:
                    found.add(f"{target}.{alias.name}")
    return found


def _trendbot_import_closure(root: str) -> set[str]:
    seen: set[str] = set()
    queue = [root]
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(_direct_trendbot_imports(current) - seen)
    return seen


def test_backtest_engine_holds_the_same_signal_function_object():
    assert backtest_module.trend_signal is signal_module.trend_signal, (
        "trendbot.engine.backtest.trend_signal is a different object from "
        "trendbot.signal.trend_signal — the engine has its own copy of the strategy."
    )
    assert signal_module.trend_signal.__module__ == "trendbot.signal"


def test_run_backtest_calls_the_module_level_signal_name():
    tree = _parse(BACKTEST_PATH)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_backtest"
    ]
    assert len(functions) == 1, "expected exactly one run_backtest in the engine"

    called_names = {
        node.func.id
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "trend_signal" in called_names, (
        "run_backtest does not call the bare name 'trend_signal'; the engine may be "
        "computing the signal some other way, which would make the identity assertion "
        "in test_backtest_engine_holds_the_same_signal_function_object vacuous."
    )


def test_engine_routes_its_signal_through_the_shared_function(monkeypatch, cfg, synth):
    real = signal_module.trend_signal
    calls: list[tuple[int, bool]] = []

    def spy(prices, lookback_days, *, long_only):
        calls.append((lookback_days, long_only))
        return real(prices, lookback_days, long_only=long_only)

    monkeypatch.setattr(backtest_module, "trend_signal", spy)
    result = backtest_module.run_backtest(synth, cfg)

    assert calls == [(cfg.lookback_days, cfg.long_only)], (
        f"expected exactly one call to the shared signal with the pre-registered "
        f"lookback {cfg.lookback_days} and long_only={cfg.long_only}, got {calls}"
    )
    assert len(result.equity) == len(synth.close)


def test_signal_module_bytes_match_the_pre_registered_hash():
    digest = hashlib.sha256(SIGNAL_PATH.read_bytes()).hexdigest()
    assert digest == SIGNAL_SHA256, f"{_HASH_FAILURE_NOTE}\n\nexpected {SIGNAL_SHA256}\nactual   {digest}"


def test_signal_module_declares_its_identity_string():
    assert signal_module.SIGNAL_ID
    assert "sign(" in signal_module.SIGNAL_ID


def test_exactly_one_trend_signal_definition_in_the_repository():
    definitions: list[str] = []
    for path in _repo_python_files():
        for node in ast.walk(_parse(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "trend_signal":
                definitions.append(f"{_rel(path)}:{node.lineno} (def)")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "trend_signal":
                        definitions.append(f"{_rel(path)}:{node.lineno} (assignment)")

    assert len(definitions) == 1, (
        "the trend signal must be defined in exactly one place; found "
        f"{len(definitions)} definitions: {definitions}"
    )
    assert definitions[0].startswith("trendbot/signal.py:"), (
        f"trend_signal is defined in {definitions[0]}, not in trendbot/signal.py"
    )


def test_no_second_momentum_sign_implementation_outside_signal_py():
    offenders: list[str] = []
    for path in _repo_python_files():
        if path == SIGNAL_PATH:
            continue
        sign_lines: list[int] = []
        shift_lines: list[int] = []
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
            if name == "sign":
                sign_lines.append(node.lineno)
            elif name == "shift":
                shift_lines.append(node.lineno)
        if sign_lines and shift_lines:
            offenders.append(f"{_rel(path)}: sign() at {sign_lines}, shift() at {shift_lines}")

    assert not offenders, (
        "a module outside trendbot/signal.py both takes a sign and shifts a series, "
        "which is what a duplicated momentum rule looks like:\n  " + "\n  ".join(offenders)
    )


def test_backtest_import_closure_reaches_trendbot_signal():
    assert "trendbot.signal" in _trendbot_import_closure("trendbot.engine.backtest")


def test_live_runner_reaches_the_same_signal_object():
    if not RUNNER_PATH.is_file():
        pytest.skip(
            "trendbot/runner.py does not exist yet (build step 7); the live half of "
            "hard invariant 2 cannot be checked until it does."
        )

    closure = _trendbot_import_closure("trendbot.runner")
    assert "trendbot.signal" in closure, (
        "trendbot.runner's static import closure never reaches trendbot.signal, so "
        "the live path is not using THE signal. Closure was: " + ", ".join(sorted(closure))
    )

    runner = importlib.import_module("trendbot.runner")

    if hasattr(runner, "trend_signal"):
        assert runner.trend_signal is signal_module.trend_signal, (
            "trendbot.runner.trend_signal is not trendbot.signal.trend_signal — the "
            "live path has its own copy of the strategy."
        )
        called_names = {
            node.func.id
            for node in ast.walk(_parse(RUNNER_PATH))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "trend_signal" in called_names, (
            "trendbot/runner.py imports trend_signal but never calls it; the live path "
            "must compute its signal with the same function the backtest uses."
        )

    divergent = {
        name: getattr(module, "trend_signal")
        for name, module in list(sys.modules.items())
        if name.startswith("trendbot")
        and module is not None
        and getattr(module, "trend_signal", None) is not None
        and getattr(module, "trend_signal") is not signal_module.trend_signal
    }
    assert not divergent, (
        "these loaded trendbot modules expose a 'trend_signal' that is not "
        f"trendbot.signal.trend_signal: {sorted(divergent)}"
    )
