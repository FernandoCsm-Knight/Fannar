"""Griffe extension for the documentation build (registered in mkdocs.yml).

The docstrings use Sphinx roles (``:class:`KernelPipeline```, ``:meth:`~fannar.X.y```), which
Markdown would print verbatim; this turns each one into inline code with the short name.
"""

import re

from griffe import Extension

ROLE = re.compile(r":(?:class|meth|func|attr|mod|data|obj):`~?([^`]+)`")


def _short(match: re.Match) -> str:
    target = match.group(1)
    return f"`{target.rsplit('.', 1)[-1] if match.group(0).count('~') else target}`"


class RstRoles(Extension):
    def on_instance(self, *, obj, **kwargs) -> None:
        if obj.docstring is not None:
            obj.docstring.value = ROLE.sub(_short, obj.docstring.value)
