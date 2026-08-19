"""Static enforcement of the hard invariants and the "Do not" list in BUILD_PROMPT.md.

Everything here inspects the source tree rather than running it, because these are
properties of the repository, not of a single execution. Several of the tests police
files that other parts of the build create (``trendbot/brokers/alpaca.py``,
``trendbot/runner.py``, the rest of ``tests/``); those skip cleanly while the file is
absent and start biting the moment it appears.

Covered here:

* invariant 4 — paper only, no live Alpaca endpoint, ``paper=True`` hardcoded;
* invariant 6 — the test suite makes no network call and imports no network client;
* invariant 1's static half — no negative shift anywhere in the package;
* "Do not use ``try/except: pass`` anywhere in the execution path";
* "Do not edit PREREGISTRATION.md" — checked against git HEAD;
* "Do not tune any parameter" — no strategy number is hardcoded in the modules that
  would be tempted to hold one, and the universe is not inlined outside the parser;
* configuration and result objects are frozen, so no parameter can be mutated after
  it is parsed.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path
from types import MappingProxyType

import pytest

from trendbot.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "trendbot"
SCRIPTS_DIR = REPO_ROOT / "scripts"
TESTS_DIR = REPO_ROOT / "tests"
SIGNAL_PATH = PACKAGE_DIR / "signal.py"
SIZING_PATH = PACKAGE_DIR / "sizing.py"
CONFIG_PATH = PACKAGE_DIR / "config.py"
ALPACA_PATH = PACKAGE_DIR / "brokers" / "alpaca.py"

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
    }
)


def _python_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
                continue
            files.append(path)
    return files


def _parse(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:  # a half-written module must not slip past the sweep
        pytest.fail(f"{path.relative_to(REPO_ROOT)} does not parse: {exc}")


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """``id()`` of every Constant node that is a module/class/function docstring."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            ids.add(id(body[0].value))
    return ids


def _numeric_literals(tree: ast.AST) -> list[tuple[float, int]]:
    """Numeric constants, skipping anything inside a slice (``xs[:5]`` is not a parameter)."""
    slice_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            for child in ast.walk(node.slice):
                slice_nodes.add(id(child))

    out: list[tuple[float, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
            and id(node) not in slice_nodes
        ):
            out.append((node.value, node.lineno))
    return out


# --------------------------------------------------------------------------------------
# invariant 4 — paper only
# --------------------------------------------------------------------------------------

_ALPACA_HOST = "api." + "alpaca.markets"  # split so this file is not itself a false hit
_PAPER_HOST = "paper-" + _ALPACA_HOST


def _live_endpoint_hits(text: str) -> list[int]:
    """Offsets of ``api.alpaca.markets`` that are NOT prefixed with ``paper-``."""
    hits = []
    for match in re.finditer(re.escape(_ALPACA_HOST), text):
        start = match.start()
        if text[max(0, start - len("paper-")) : start] != "paper-":
            hits.append(start)
    return hits


def test_alpaca_adapter_uses_only_the_paper_endpoint():
    if not ALPACA_PATH.is_file():
        pytest.skip("trendbot/brokers/alpaca.py does not exist yet (build step 6)")
    text = ALPACA_PATH.read_text(encoding="utf-8")

    bad = _live_endpoint_hits(text)
    assert not bad, (
        f"{_rel(ALPACA_PATH)} references the LIVE trading host at offsets {bad}. "
        "BUILD_PROMPT.md invariant 4: paper only, no live code path."
    )
    assert _PAPER_HOST in text, (
        f"{_rel(ALPACA_PATH)} never names {_PAPER_HOST}; the paper endpoint must be "
        "explicit in the adapter, not assembled at runtime from configuration."
    )


def test_alpaca_adapter_hardcodes_paper_true():
    if not ALPACA_PATH.is_file():
        pytest.skip("trendbot/brokers/alpaca.py does not exist yet (build step 6)")
    tree = _parse(ALPACA_PATH)

    hardcoded: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg and "paper" in node.arg.lower():
            if isinstance(node.value, ast.Constant) and node.value.value is True:
                hardcoded.append(f"keyword {node.arg}=True")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and "paper" in target.id.lower()
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True
                ):
                    hardcoded.append(f"{target.id} = True (line {node.lineno})")
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and "paper" in node.target.id.lower()
                and isinstance(node.value, ast.Constant)
                and node.value.value is True
            ):
                hardcoded.append(f"{node.target.id} = True (line {node.lineno})")

    assert hardcoded, (
        f"{_rel(ALPACA_PATH)} contains no literal paper=True. Invariant 4 requires the "
        "paper flag to be hardcoded, not read from config, so that no configuration "
        "change can turn this adapter live."
    )


def test_no_live_alpaca_endpoint_anywhere_in_the_repository():
    """Invariant 4 applies to the whole repo, not just the adapter."""
    offenders: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".md", ".toml", ".cfg", ".ini", ".json"}:
            continue
        rel = path.relative_to(REPO_ROOT)
        if any(part in _SKIP_DIRS for part in rel.parts) or rel.parts[0] == "tests":
            continue
        if _live_endpoint_hits(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(rel.as_posix())
    assert not offenders, f"live Alpaca trading host referenced in: {offenders}"


def test_no_module_sets_paper_false():
    offenders: list[str] = []
    for path in _python_files(PACKAGE_DIR, SCRIPTS_DIR):
        for node in ast.walk(_parse(path)):
            if (
                isinstance(node, ast.keyword)
                and node.arg
                and "paper" in node.arg.lower()
                and isinstance(node.value, ast.Constant)
                and node.value.value is False
            ):
                offenders.append(f"{_rel(path)}:{getattr(node.value, 'lineno', '?')}")
    assert not offenders, f"paper=False appears in: {offenders}"


# --------------------------------------------------------------------------------------
# invariant 6 — offline tests
# --------------------------------------------------------------------------------------

_FORBIDDEN_IMPORT_ROOTS = frozenset({"requests", "yfinance", "socket", "urllib3", "httpx", "aiohttp"})
_FORBIDDEN_SUBMODULES = frozenset({"urllib.request", "urllib.error", "http.client"})
_NETWORK_LOADERS = frozenset({"load_prices", "load_risk_free_rate", "last_trade_prices"})


def _test_suite_files() -> list[Path]:
    files = _python_files(TESTS_DIR)
    assert files, "no test files found — the offline sweep would be vacuous"
    return files


def test_test_suite_imports_no_network_client():
    offenders: list[str] = []
    for path in _test_suite_files():
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _FORBIDDEN_IMPORT_ROOTS or alias.name in _FORBIDDEN_SUBMODULES:
                        offenders.append(f"{_rel(path)}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                if root in _FORBIDDEN_IMPORT_ROOTS or module in _FORBIDDEN_SUBMODULES:
                    offenders.append(f"{_rel(path)}:{node.lineno} from {module}")
                for alias in node.names:
                    if f"{module}.{alias.name}" in _FORBIDDEN_SUBMODULES:
                        offenders.append(f"{_rel(path)}:{node.lineno} from {module} import {alias.name}")

    assert not offenders, (
        "BUILD_PROMPT.md invariant 6: the test suite must run with the machine "
        "offline, so no test may import a network client:\n  " + "\n  ".join(offenders)
    )


def test_test_suite_never_calls_a_network_data_loader():
    offenders: list[str] = []
    for path in _test_suite_files():
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("trendbot"):
                for alias in node.names:
                    if alias.name in _NETWORK_LOADERS:
                        offenders.append(f"{_rel(path)}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else func.id
                    if isinstance(func, ast.Name)
                    else ""
                )
                if name in _NETWORK_LOADERS:
                    offenders.append(f"{_rel(path)}:{node.lineno} calls {name}()")

    assert not offenders, (
        "these tests reach for the live price vendors; fixtures or "
        "trendbot.engine.validation.synthetic_prices must be used instead:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------------------
# "Do not use try/except: pass anywhere in the execution path"
# --------------------------------------------------------------------------------------


def _is_noop(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Continue):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
        return statement.value.value is Ellipsis
    return False


def test_no_exception_is_silently_swallowed():
    offenders: list[str] = []
    for path in _python_files(PACKAGE_DIR, SCRIPTS_DIR):
        for node in ast.walk(_parse(path)):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.body and all(_is_noop(statement) for statement in node.body):
                caught = ast.unparse(node.type) if node.type else "BaseException (bare except)"
                offenders.append(f"{_rel(path)}:{node.lineno} except {caught}: <no-op>")

    assert not offenders, (
        "BUILD_PROMPT.md: 'Do not use try/except: pass anywhere in the execution "
        "path.' A swallowed exception in a trading loop is how a halt condition "
        "becomes a silent no-op:\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------------------
# invariant 1's static half — no negative shift in the package
# --------------------------------------------------------------------------------------


def test_no_negative_shift_in_the_package():
    """A ``.shift(-n)`` outside the tests is a lookahead by construction."""
    offenders: list[str] = []
    for path in _python_files(PACKAGE_DIR, SCRIPTS_DIR):
        for node in ast.walk(_parse(path)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in {"shift", "tshift"}:
                continue
            for argument in list(node.args) + [kw.value for kw in node.keywords]:
                if (
                    isinstance(argument, ast.UnaryOp)
                    and isinstance(argument.op, ast.USub)
                    and isinstance(argument.operand, ast.Constant)
                ):
                    offenders.append(f"{_rel(path)}:{node.lineno} {ast.unparse(node)[:80]}")
    assert not offenders, (
        "negative shift in the execution path — hard invariant 1 says information "
        "only ever moves forward:\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------------------
# "Do not edit PREREGISTRATION.md"
# --------------------------------------------------------------------------------------


def test_preregistration_is_unmodified_relative_to_git_head():
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not on PATH")
    inside = subprocess.run(
        [git, "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip("not inside a git work tree")

    proc = subprocess.run(
        [git, "diff", "--quiet", "HEAD", "--", "PREREGISTRATION.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "PREREGISTRATION.md differs from git HEAD. BUILD_PROMPT.md: 'Do not edit "
        "PREREGISTRATION.md.' Editing it after seeing a result is the failure the "
        f"document exists to prevent.\n{proc.stderr.strip()}\n"
        "Run: git diff HEAD -- PREREGISTRATION.md"
    )


def test_config_hash_matches_the_document_on_disk():
    """The config carries the sha256 of the text it was actually parsed from."""
    import hashlib

    cfg = load_config()
    on_disk = hashlib.sha256((REPO_ROOT / "PREREGISTRATION.md").read_bytes()).hexdigest()
    assert cfg.source_sha256 == on_disk
    assert cfg.source_path == REPO_ROOT / "PREREGISTRATION.md"


# --------------------------------------------------------------------------------------
# "Do not tune any parameter" — no strategy number inlined in the strategy modules
# --------------------------------------------------------------------------------------


def test_signal_module_contains_no_numeric_literal_but_zero_one_and_tolerances():
    """The lookback must arrive as an argument, never as a 252 in the source.

    Anything other than 0, 1 or a small tolerance in trendbot/signal.py is a
    strategy number that has escaped PREREGISTRATION.md.
    """
    bad = [
        (value, line)
        for value, line in _numeric_literals(_parse(SIGNAL_PATH))
        if value not in (0, 1) and not (0 < abs(value) <= 1e-3)
    ]
    assert not bad, (
        "trendbot/signal.py contains numeric literals that are neither 0, 1 nor a "
        f"tolerance: {bad}. Every strategy parameter must come from Config, which "
        "parses PREREGISTRATION.md."
    )


def test_sizing_uses_252_only_to_annualise():
    """``sqrt(252)`` is legitimate; a 252 used as a lookback is not."""
    tree = _parse(SIZING_PATH)

    annualisation_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and re.search(r"TRADING_DAYS|ANNUAL", target.id, re.IGNORECASE)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 252
    }
    assert annualisation_names, "expected a named annualisation constant in trendbot/sizing.py"

    stray = [line for value, line in _numeric_literals(tree) if value == 252 and line not in
             {node.value.lineno for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Constant) and node.value.value == 252}]
    assert not stray, (
        f"trendbot/sizing.py uses the literal 252 outside its annualisation constant, "
        f"at line(s) {stray}. If that is the lookback it must come from Config."
    )

    # ...and the constant itself is only ever used to annualise: sqrt(x) or a product.
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node
    misuse: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and node.id in annualisation_names):
            continue
        if isinstance(node.ctx, ast.Store):
            continue
        owner = parent.get(id(node))
        annualising = (isinstance(owner, ast.BinOp) and isinstance(owner.op, (ast.Mult, ast.Div))) or (
            isinstance(owner, ast.Call)
            and isinstance(owner.func, ast.Attribute)
            and owner.func.attr in {"sqrt", "power"}
        )
        if not annualising:
            misuse.append(f"line {node.lineno}")
    assert not misuse, (
        f"{annualisation_names} is used for something other than annualisation at {misuse}; "
        "252 is also the pre-registered lookback and the two must not be conflated."
    )


def test_no_pre_registered_parameter_is_hardcoded_in_signal_or_sizing():
    cfg = load_config()
    forbidden = {
        float(cfg.lookback_days): "lookback_days",
        float(cfg.ewma_halflife_days): "ewma_halflife_days",
        float(cfg.n_universe): "n_universe",
        cfg.instrument_vol_target: "instrument_vol_target",
        cfg.portfolio_vol_target: "portfolio_vol_target",
        cfg.per_instrument_cap: "per_instrument_cap",
        cfg.drift_band: "drift_band",
        cfg.cost_bps_per_side: "cost_bps_per_side",
        cfg.cost_rate_per_side: "cost_rate_per_side",
    }
    # 252 in sizing.py is the annualisation factor, covered precisely by the test above.
    exempt = {SIZING_PATH: {float(cfg.lookback_days)}}

    offenders: list[str] = []
    for path in (SIGNAL_PATH, SIZING_PATH):
        allowed = exempt.get(path, set())
        for value, line in _numeric_literals(_parse(path)):
            if float(value) in forbidden and float(value) not in allowed:
                offenders.append(f"{_rel(path)}:{line} literal {value} == cfg.{forbidden[float(value)]}")

    assert not offenders, (
        "a pre-registered parameter is inlined instead of being read from Config:\n  "
        + "\n  ".join(offenders)
    )


def test_universe_tickers_are_not_inlined_outside_the_config_parser():
    """Section 2's universe lives in PREREGISTRATION.md; only the parser may name it.

    Docstrings are exempt — prose that mentions SPY's 1993 inception is documentation,
    not a hardcoded universe.
    """
    cfg = load_config()
    patterns = {ticker: re.compile(rf"\b{re.escape(ticker)}\b") for ticker in cfg.universe}

    offenders: list[str] = []
    for path in _python_files(PACKAGE_DIR, SCRIPTS_DIR):
        if path == CONFIG_PATH:
            continue
        tree = _parse(path)
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                for ticker, pattern in patterns.items():
                    if pattern.search(node.value):
                        offenders.append(f"{_rel(path)}:{node.lineno} string contains {ticker!r}")
            elif isinstance(node, ast.Name) and node.id in patterns:
                offenders.append(f"{_rel(path)}:{node.lineno} identifier {node.id}")

    assert not offenders, (
        "the universe is fixed by PREREGISTRATION.md section 2 and must be read "
        "through Config.universe, never inlined:\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------------------
# configuration and result objects are immutable
# --------------------------------------------------------------------------------------


def _is_dataclass_decorator(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id == "dataclass"
    if isinstance(target, ast.Attribute):
        return target.attr == "dataclass"
    return False


def _declares_frozen(decorators: list[ast.expr]) -> bool:
    return any(
        isinstance(d, ast.Call)
        and any(
            kw.arg == "frozen" and isinstance(kw.value, ast.Constant) and kw.value.value is True
            for kw in d.keywords
        )
        for d in decorators
    )


def _mutates_own_attributes(node: ast.ClassDef) -> bool:
    """True when a method of the class assigns to ``self.<attr>``.

    Such a class is stateful by construction and could not be frozen even in
    principle; a value object that never rebinds its own fields could be, and
    therefore must be.
    """
    for member in node.body:
        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for statement in ast.walk(member):
            targets: list[ast.expr] = []
            if isinstance(statement, ast.Assign):
                targets = list(statement.targets)
            elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
                targets = [statement.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    return True
    return False


def _package_dataclasses() -> list[tuple[Path, ast.ClassDef]]:
    found = []
    for path in _python_files(PACKAGE_DIR):
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ClassDef) and any(
                _is_dataclass_decorator(d) for d in node.decorator_list
            ):
                found.append((path, node))
    assert found, "no dataclasses found in the package — the sweep would be vacuous"
    return found


def test_public_value_dataclasses_are_frozen():
    """A parsed parameter or a recorded result must not be mutable after the fact.

    The only exemption is structural rather than declarative: a dataclass that
    rebinds its own attributes is runtime state and cannot be frozen. Everything
    else in this package is a config or a result and has to say so.
    """
    offenders: list[str] = []
    for path, node in _package_dataclasses():
        if node.name.startswith("_") or _mutates_own_attributes(node):
            continue
        decorators = [d for d in node.decorator_list if _is_dataclass_decorator(d)]
        if not _declares_frozen(decorators):
            offenders.append(f"{_rel(path)}:{node.lineno} {node.name}")

    assert not offenders, (
        "these public dataclasses never mutate themselves, so they are value objects "
        "and must be declared frozen; a config or a result that can be edited after "
        "it is produced defeats pre-registration:\n  " + "\n  ".join(offenders)
    )


def test_every_dataclass_returned_by_the_package_is_frozen():
    """No exemption here: anything a function hands back is a result.

    Complements the test above, which lets self-mutating runtime state opt out.
    A type that appears in a return annotation is being published as a result and
    must be immutable however it is implemented.
    """
    dataclasses_by_name = {node.name: (path, node) for path, node in _package_dataclasses()}

    returned: set[str] = set()
    for path in _python_files(PACKAGE_DIR):
        for node in ast.walk(_parse(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
                for child in ast.walk(node.returns):
                    if isinstance(child, ast.Name) and child.id in dataclasses_by_name:
                        returned.add(child.id)
                    elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                        returned.update(
                            name for name in dataclasses_by_name if re.fullmatch(rf"\s*{name}\s*", child.value)
                        )

    assert returned, "no dataclass appears in a return annotation — the sweep would be vacuous"
    offenders = [
        f"{_rel(dataclasses_by_name[name][0])}:{dataclasses_by_name[name][1].lineno} {name}"
        for name in sorted(returned)
        if not _declares_frozen(
            [d for d in dataclasses_by_name[name][1].decorator_list if _is_dataclass_decorator(d)]
        )
    ]
    assert not offenders, "these result types are returned by the package but are not frozen:\n  " + "\n  ".join(
        offenders
    )


def test_config_object_rejects_mutation(cfg):
    with pytest.raises(Exception):
        cfg.lookback_days = 200  # type: ignore[misc]
    assert cfg.lookback_days == load_config().lookback_days


_IMMUTABLE = (str, bytes, int, float, bool, type(None), tuple, frozenset, Path, MappingProxyType)


def _deeply_immutable(value: object) -> bool:
    if isinstance(value, tuple):
        return all(_deeply_immutable(item) for item in value)
    if isinstance(value, MappingProxyType):
        return all(_deeply_immutable(item) for item in value.values())
    return isinstance(value, _IMMUTABLE)


def test_frozen_config_holds_no_mutable_parameter():
    """`frozen=True` is not immutability if a field holds a dict.

    Uses an uncached parse so that this test cannot itself corrupt the shared
    session config — which is exactly the accident a mutable field invites.
    """
    import dataclasses

    cfg = load_config(use_cache=False)
    mutable = {
        field.name: type(getattr(cfg, field.name)).__name__
        for field in dataclasses.fields(cfg)
        if not _deeply_immutable(getattr(cfg, field.name))
    }
    assert not mutable, (
        "these Config fields hold mutable objects, so a pre-registered parameter can "
        f"be edited at runtime despite frozen=True: {mutable}"
    )
