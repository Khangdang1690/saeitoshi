"""Command-line interface for saeitoshi.

Registered as ``saeitoshi`` in ``pyproject.toml`` ``[project.scripts]``.
Implemented in M4 once the Python surface lands.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    print("saeitoshi CLI: not yet implemented (lands in M4)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
