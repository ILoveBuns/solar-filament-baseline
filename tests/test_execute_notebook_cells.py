from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.execute_notebook_cells import execute


class ExecuteNotebookCellsTest(unittest.TestCase):
    def test_executes_code_cells_in_one_namespace(self):
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["ignored"]},
                {"cell_type": "code", "source": ["value = 41\n"]},
                {"cell_type": "code", "source": ["print(value + 1)\n"]},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.ipynb"
            path.write_text(json.dumps(notebook))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                count = execute(path)
        self.assertEqual(2, count)
        self.assertIn("42", output.getvalue())
        self.assertIn("EXECUTED_CODE_CELLS=2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
