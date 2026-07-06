"""Execution services that do not depend on PyQt or Flask."""

from __future__ import annotations

import ast
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable


@dataclass(slots=True)
class RunResult:
    """Outcome of executing a generated SURF script."""

    success: bool
    message: str
    output: str
    model: Any = None


class _BeforeModelSolve(ast.NodeTransformer):
    """Insert a progress callback immediately before ``model.solve(...)``."""

    @staticmethod
    def _is_model_solve(statement: ast.stmt) -> bool:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return False
        function = statement.value.func
        return (
            isinstance(function, ast.Attribute)
            and function.attr == "solve"
            and isinstance(function.value, ast.Name)
            and function.value.id == "model"
        )

    def visit_Module(self, node: ast.Module) -> ast.Module:
        statements: list[ast.stmt] = []
        for statement in node.body:
            if self._is_model_solve(statement):
                statements.append(
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="__surf_before_solve__", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        )
                    )
                )
            statements.append(statement)
        node.body = statements
        return node


def run_generated_code(
    code_text: str,
    before_solve: Callable[[], None] | None = None,
) -> RunResult:
    """Execute generated SURF code and capture its model and terminal output."""
    output_stream = StringIO()
    try:
        namespace: dict[str, Any] = {"__surf_before_solve__": before_solve or (lambda: None)}
        syntax_tree = ast.parse(code_text, filename="<generated-surf-script>")
        syntax_tree = ast.fix_missing_locations(_BeforeModelSolve().visit(syntax_tree))
        with redirect_stdout(output_stream), redirect_stderr(output_stream):
            exec(compile(syntax_tree, "<generated-surf-script>", "exec"), namespace)
        return RunResult(
            success=True,
            message="SURF run completed successfully.",
            output=output_stream.getvalue(),
            model=namespace.get("model"),
        )
    except Exception:
        output_stream.write(traceback.format_exc())
        return RunResult(
            success=False,
            message="SURF run failed.",
            output=output_stream.getvalue(),
        )
