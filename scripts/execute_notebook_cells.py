from __future__ import annotations

import argparse
import json
from pathlib import Path


def execute(notebook_path: Path) -> int:
    """Execute trusted notebook code cells sequentially in one namespace."""
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    executed = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        print(f"EXECUTING CELL {index}", flush=True)
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}#cell-{index}", "exec"), namespace)
        executed += 1
    print(f"EXECUTED_CODE_CELLS={executed}", flush=True)
    return executed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequentially execute every code cell in a trusted notebook"
    )
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    execute(args.notebook)


if __name__ == "__main__":
    main()
