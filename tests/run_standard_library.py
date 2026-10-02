"""Run the exact test functions when pytest is unavailable.

Normal command: python -m pytest. Fallback (after installing mc_pricer):
python tests/run_standard_library.py

This uses Python's actual unittest runner, not a replacement pytest module.
"""

import argparse
import json
from pathlib import Path
import platform
import runpy
import sys
import unittest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-name", default="test_results.json")
    args = parser.parse_args()
    if Path(args.report_name).name != args.report_name:
        parser.error("--report-name must be a filename without directories")
    root = Path(__file__).resolve().parents[1]
    # Experiment modules live outside the installed pricing package.
    sys.path.insert(0, str(root))
    tests = []
    test_names = []
    for source in sorted((root / "tests").glob("test_*.py")):
        namespace = runpy.run_path(str(source))
        for name, value in sorted(namespace.items()):
            if name.startswith("test_") and callable(value):
                tests.append(value)
                test_names.append(f"{source.name}::{name}")
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "runner": "unittest.FunctionTestCase",
        "python": platform.python_version(),
        "test_functions_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "test_functions": test_names,
        "pytest_executed": False,
        "note": "The same plain test functions are discoverable by pytest. "
                "This report records standard-library execution only.",
    }
    destination = root / "results" / "raw" / args.report_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
