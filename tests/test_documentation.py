"""Standard-library documentation checks; never download data or train models."""

import ast
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from importlib.metadata import PackageNotFoundError
from unittest.mock import Mock, patch

from generate_documentation import (
    END, METRICS, ROOT, START, cell_source, code_tree, generate,
    notebook_details, render, replace_results, result_table,
)


class DocumentationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("ser_professional_colab.ipynb", "build_professional_notebook.py",
                     "requirements.txt", "generate_documentation.py", "README.md"):
            shutil.copyfile(ROOT / name, self.root / name)
        self.models = notebook_details(self.root)[5]

    def update_notebook(self, old, new):
        path = self.root / "ser_professional_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for cell in notebook["cells"]:
            source = cell_source(cell)
            if old in source:
                cell["source"] = source.replace(old, new)
                changed = True
        self.assertTrue(changed, f"Test source missing: {old}")
        path.write_text(json.dumps(notebook), encoding="utf-8")

    def write_result_fixture(self):
        # Deliberately synthetic values, only inside a temporary test directory.
        path = self.root / "results" / "model_comparison.csv"
        path.parent.mkdir()
        fields = ["model", "parameters", "best_epoch", *METRICS]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for model in self.models:
                writer.writerow({"model": model, "parameters": 64, "best_epoch": 1,
                                 **dict.fromkeys(METRICS, 0.5)})
        return path

    def test_notebook_code_is_parseable_without_executing(self):
        notebook = json.loads((self.root / "ser_professional_colab.ipynb").read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(code_tree(cell_source(cell)), f"<notebook cell {index + 1}>", "exec")

    def test_current_contracts_and_sources(self):
        cells, config, labels, datasets, features, models, model_code, synced = notebook_details(self.root)
        self.assertGreater(len(cells), len(models))
        self.assertEqual(config["sample_rate"], 16000)
        self.assertEqual(labels, ["angry", "disgust", "fear", "happy", "neutral", "sad"])
        self.assertEqual(set(datasets), {"ravdess", "cremad", "tess", "savee"})
        n = config["n_mfcc"]
        self.assertEqual([len(columns) for columns in features.values()], [n, 3 * n, 3 * n + 2, 3 * n + 5])
        self.assertEqual(models, ("lstm", "bilstm", "cnn_bilstm"))
        self.assertIn("layers.Bidirectional", model_code)
        self.assertTrue(synced)

    def test_deterministic_generation_and_read_only_freshness_check(self):
        changed = generate(self.root)
        self.assertIn("docs/PROJECT_GUIDE.md", changed)
        before = render(self.root)
        self.assertEqual(generate(self.root, check=True), [])
        self.assertEqual(generate(self.root), [])
        self.assertEqual(before, render(self.root))
        (self.root / "docs" / "PROJECT_GUIDE.md").write_text("stale", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Outdated documentation"):
            generate(self.root, check=True)
        self.assertEqual((self.root / "docs" / "PROJECT_GUIDE.md").read_text(), "stale")

    def test_config_updates_dimensions_and_warns_on_unsynced_generator(self):
        generate(self.root)
        self.update_notebook("duration: float = 4.0", "duration: float = 3.0")
        with self.assertRaisesRegex(ValueError, "Outdated documentation"):
            generate(self.root, check=True)
        generate(self.root)
        guide = (self.root / "docs" / "PROJECT_GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("| `duration` | `3.0` |", guide)
        self.assertIn("`(301, selected_channels)`", guide)
        self.assertIn("DIFFERENT: synchronize", guide)

    def test_model_builder_changes_appear_without_training(self):
        self.update_notebook("layers.LSTM(64", "layers.LSTM(48")
        guide = render(self.root)["docs/PROJECT_GUIDE.md"]
        self.assertIn("layers.LSTM(48", guide)

    def test_requirements_changes_refresh_guide(self):
        before = render(self.root)["docs/PROJECT_GUIDE.md"]
        path = self.root / "requirements.txt"
        path.write_text(path.read_text(encoding="utf-8") + "\n# test-only source change\n", encoding="utf-8")
        after = render(self.root)["docs/PROJECT_GUIDE.md"]
        self.assertNotEqual(before, after)
        self.assertIn("# test-only source change", after)

    def test_readme_prose_outside_results_is_preserved(self):
        original = f"Custom introduction\n{START}\nold\n{END}\nCustom closing\n"
        expected = f"Custom introduction\n{START}\nnew\n{END}\nCustom closing\n"
        self.assertEqual(replace_results(original, "new"), expected)
        for broken in ("no markers", START + END + START, END + START):
            with self.assertRaises(ValueError):
                replace_results(broken, "new")

    def test_absent_results_never_get_performance_values(self):
        table, present = result_table(self.root, self.models)
        self.assertFalse(present)
        self.assertEqual(table.count("Run required"), len(self.models) * 7)
        self.assertIn("No completed experiment table", render(self.root)["docs/RESULTS.md"])

    def test_csv_fixture_is_used_only_when_present(self):
        self.write_result_fixture()
        table, present = result_table(self.root, self.models)
        self.assertTrue(present)
        self.assertIn("| LSTM | 64 | 1 | 0.5000", table)
        self.assertNotIn("Run required", table)

    def test_invalid_results_fail_instead_of_printing_scores(self):
        path = self.write_result_fixture()
        original = path.read_text(encoding="utf-8")
        for invalid in ("nan", "inf", "-0.1", "1.1", "not-a-number"):
            with self.subTest(invalid=invalid):
                path.write_text(original.replace("0.5", invalid, 1), encoding="utf-8")
                with self.assertRaises(ValueError):
                    result_table(self.root, self.models)
        for invalid in ("0", "-4", "1.5", "nan"):
            with self.subTest(parameters=invalid):
                path.write_text(original.replace(",64,", f",{invalid},", 1), encoding="utf-8")
                with self.assertRaises(ValueError):
                    result_table(self.root, self.models)
        path.write_text(original.replace("cnn_bilstm", "lstm"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            result_table(self.root, self.models)
        path.write_text("model,accuracy\nlstm,0.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema"):
            result_table(self.root, self.models)

    def test_colab_secrets_and_setup_remain_documented(self):
        guide = render(self.root)["docs/PROJECT_GUIDE.md"]
        self.assertIn("google.colab.userdata.get()", guide)
        self.assertIn("No local `kaggle.json` is required", guide)
        self.assertIn('"gradio>=6,<7"', guide)
        self.assertIn("| 1 | 3 | 1. Environment", guide)
        self.assertIn("| 3 | 6 | 2. Reproducible dataset", guide)

    def test_missing_jedi_recovers_only_in_colab(self):
        notebook = json.loads((self.root / "ser_professional_colab.ipynb").read_text(encoding="utf-8"))
        imports_tree = code_tree(cell_source(notebook["cells"][3]))
        definition = next(node for node in imports_tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == "ensure_jedi")
        run = Mock()
        namespace = {"version": Mock(side_effect=[PackageNotFoundError("jedi"), "0.19.2"]),
                     "PackageNotFoundError": PackageNotFoundError,
                     "subprocess": types.SimpleNamespace(run=run, CalledProcessError=subprocess.CalledProcessError),
                     "sys": sys}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "<Colab dependency check>", "exec"), namespace)
        google = types.ModuleType("google")
        google.colab = types.ModuleType("google.colab")
        with patch.dict(sys.modules, {"google": google, "google.colab": google.colab}):
            self.assertEqual(namespace["ensure_jedi"](), "0.19.2")
        run.assert_called_once_with([sys.executable, "-m", "pip", "install", "-q", "jedi>=0.19,<1"], check=True)

        namespace["version"] = Mock(return_value="0.19.2")
        run.reset_mock()
        self.assertEqual(namespace["ensure_jedi"](), "0.19.2")
        run.assert_not_called()

    def test_missing_csv_values_have_actionable_errors(self):
        path = self.write_result_fixture()
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace("0.5", "", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing or invalid numeric result"):
            result_table(self.root, self.models)
        path.write_text(original.replace(",0.5,0.5,0.5,0.5,0.5", "", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing or invalid numeric result"):
            result_table(self.root, self.models)


if __name__ == "__main__":
    unittest.main()
