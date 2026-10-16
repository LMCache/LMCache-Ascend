# SPDX-License-Identifier: Apache-2.0
"""Isolate production host logic without importing the NPU plugin bootstrap.

No implementation is copied into these fixtures. Full small modules or selected
production class methods are compiled from their real source files. Device and
framework dependencies are explicit test doubles; this is not device coverage.
"""

# Standard
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import ast
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[2]


def selected_class(path, name, methods, namespace, parent=object, optional=()):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    body = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef)
        and node.name in set(methods).union(optional)
    ]
    if not set(methods).issubset({node.name for node in body}):
        raise AssertionError(f"Production methods missing in {name}")
    for node in body:
        node.decorator_list = []
    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=body,
        decorator_list=[],
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    scope = {"Parent": parent, **namespace}
    exec(compile(ast.fix_missing_locations(unit), str(ROOT / path), "exec"), scope)
    return scope[name]


def package(name, **members):
    module = ModuleType(name)
    module.__path__ = []
    module.__dict__.update(members)
    return module


@contextmanager
def source_module(path, name, dependencies=None):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module, **(dependencies or {})}):
        spec.loader.exec_module(module)
        yield module
