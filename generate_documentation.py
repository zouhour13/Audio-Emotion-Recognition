"""Refresh project documentation without importing ML libraries or running cells."""

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
START = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"
METRICS = ("val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1")


def cell_source(cell):
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def code_tree(source):
    # Colab's package magic is not Python syntax. Never execute notebook code.
    return ast.parse("\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%")))


def assignment(trees, name):
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node.value
    raise ValueError(f"Notebook assignment missing: {name}")


def feature_value(node, config):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "CFG":
        return config[node.attr]
    if isinstance(node, ast.BinOp):
        left, right = feature_value(node.left, config), feature_value(node.right, config)
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
    if isinstance(node, ast.List):
        return [feature_value(value, config) for value in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [feature_value(value, config) for value in node.args]
        if node.func.id == "range":
            return range(*args)
        if node.func.id == "list":
            return list(*args)
    raise ValueError(f"Unsupported feature configuration: {ast.unparse(node)}")


def notebook_details(root):
    notebook = json.loads((root / "ser_professional_colab.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    trees = [code_tree(cell_source(cell)) for cell in cells if cell["cell_type"] == "code"]
    config_class = next(node for tree in trees for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "Config")
    config = {node.target.id: ast.literal_eval(node.value) for node in config_class.body
              if isinstance(node, ast.AnnAssign)}
    labels = ast.literal_eval(assignment(trees, "LABELS"))
    datasets = ast.literal_eval(assignment(trees, "KAGGLE_DATASETS"))
    feature_node = assignment(trees, "FEATURE_SETS")
    features = {ast.literal_eval(key): feature_value(value, config)
                for key, value in zip(feature_node.keys, feature_node.values)}
    models = None
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.For) and isinstance(node.iter, (ast.Tuple, ast.List)):
                if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                       and call.func.id == "train_model" for call in ast.walk(node)):
                    models = ast.literal_eval(node.iter)
    if not models:
        raise ValueError("Notebook model-comparison loop missing")
    model_code = next(ast.unparse(node) for tree in trees for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "make_model")
    generator = ast.parse((root / "build_professional_notebook.py").read_text(encoding="utf-8"))
    generated_cells = assignment([generator], "cells")
    sources = [ast.literal_eval(call.args[0]).strip() + "\n" for call in generated_cells.elts]
    synced = sources == [cell_source(cell) for cell in cells]
    return cells, config, labels, datasets, features, models, model_code, synced


def model_title(name):
    return {"lstm": "LSTM", "bilstm": "BiLSTM", "cnn_bilstm": "CNN-BiLSTM"}.get(name, name)


def result_number(row, field):
    try:
        return float(row[field])
    except (ValueError, TypeError):
        raise ValueError(f"Missing or invalid numeric result: {field}") from None


def result_table(root, models):
    path = root / "results" / "model_comparison.csv"
    rows = None
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"model", "parameters", "best_epoch", *METRICS}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("results/model_comparison.csv does not match the notebook's comparison schema")
            rows = list(reader)
        if len(rows) != len(models) or {row["model"] for row in rows} != set(models):
            raise ValueError("Results must contain exactly one row for every compared model")
        for row in rows:
            for field in ("parameters", "best_epoch"):
                number = result_number(row, field)
                if not math.isfinite(number) or not number.is_integer() or number <= 0:
                    raise ValueError(f"Invalid positive integer in results: {field}")
            for field in METRICS:
                number = result_number(row, field)
                if not math.isfinite(number) or not 0 <= number <= 1:
                    raise ValueError(f"Invalid metric in results: {field}; expected a fraction in [0, 1]")
    lines = ["| Model | Parameters | Best epoch | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    if rows is None:
        lines += [f"| {model_title(name)} | " + " | ".join(["Run required"] * 7) + " |" for name in models]
    else:
        for row in rows:
            values = [model_title(row["model"]), str(int(float(row["parameters"]))), str(int(float(row["best_epoch"])))]
            values += [f"{float(row[field]):.4f}" for field in METRICS]
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines), rows is not None


def replace_results(readme, table):
    if readme.count(START) != 1 or readme.count(END) != 1 or readme.index(START) >= readme.index(END):
        raise ValueError("README needs exactly one ordered pair of generated-results markers")
    before, remainder = readme.split(START)
    _, after = remainder.split(END)
    return before + START + "\n" + table + "\n" + END + after


def render(root):
    cells, config, labels, datasets, features, models, model_code, synced = notebook_details(root)
    table, has_results = result_table(root, models)
    rows = []
    section, code_number = "Project objective", 0
    for number, cell in enumerate(cells, 1):
        source = cell_source(cell)
        if cell["cell_type"] == "markdown":
            headings = re.findall(r"^## (.+)$", source, re.MULTILINE)
            if headings:
                section = headings[0]
        else:
            code_number += 1
            rows.append(f"| {code_number} | {number} | {section} |")
    execution = "\n".join(["| Code cell | Notebook position | Section |", "|---:|---:|---|", *rows])
    configuration = "\n".join(["| Setting | Default |", "|---|---|",
                                  *[f"| `{key}` | `{value}` |" for key, value in config.items()]])
    corpora = "\n".join(["| Corpus | Kaggle source |", "|---|---|",
                            *[f"| {name.upper()} | [{handle}](https://www.kaggle.com/datasets/{handle}) |"
                              for name, handle in datasets.items()]])
    feature_rows = "\n".join(["| Feature candidate | Channels |", "|---|---:|",
                                 *[f"| `{name}` | {len(columns)} |" for name, columns in features.items()]])
    install = next(cell_source(cell).strip() for cell in cells if cell["cell_type"] == "code"
                   and "%pip install" in cell_source(cell))
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").strip()
    frames = 1 + int(config["sample_rate"] * config["duration"]) // config["hop_length"]
    fingerprint = hashlib.sha256()
    for name in ("ser_professional_colab.ipynb", "build_professional_notebook.py", "requirements.txt", "generate_documentation.py"):
        fingerprint.update(name.encode())
        fingerprint.update((root / name).read_text(encoding="utf-8").replace("\r\n", "\n").encode())
    guide = f"""# Project Guide

<!-- Generated by generate_documentation.py. Edit the source or docs/MAINTENANCE.md, not this file. -->

This guide is generated from the professional notebook's actual configuration, dataset handles,
feature definitions, model builder, and cell order. No notebook cells are executed to build documentation.

## Scope And Status

Six acted-speech labels, in saved model order: {', '.join(f'`{label}`' for label in labels)}.
The historical 78.57% score is methodologically compromised and is not a benchmark for this rebuild.
Documentation generation does not perform training, validate CUDA, or certify model quality.
See [results](RESULTS.md) for the current publication status.

Notebook and Python generator sources: **{'in sync' if synced else 'DIFFERENT: synchronize them before rebuilding; the guide describes the notebook'}**.
Source fingerprint: `{fingerprint.hexdigest()[:16]}`. This is a freshness identifier, not an experiment ID.

## Colab Setup

1. Open [the professional notebook](../ser_professional_colab.ipynb) in Google Colab.
2. Select a GPU runtime before running code. Confirm the imports cell reports a GPU device.
3. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to the Colab Secrets panel and enable notebook access for both.
   Use Kaggle's legacy username/API-key pair, not an email, password, or a newer token of a different format.
4. Run code cell 1 (package setup), restart the session if prompted, then run code cell 2 (imports/configuration).
5. Run code cell 3 (dataset acquisition), followed by the remaining code cells in the order below.
6. Download the artifact directory before the temporary Colab runtime is deleted.

`configure_kaggle()` reads both secrets with `google.colab.userdata.get()` and sets environment
variables before importing and authenticating `KaggleApi`. A missing or denied Colab secret raises
a clear error instead of falling back to stale credentials. No local `kaggle.json` is required.
Only outside Colab is an explicitly configured environment-variable fallback supported.
Never put credentials in notebook outputs, commits, or documentation.
If `jedi` is missing after a Colab reconnect, the imports cell installs that IPython dependency
once into the active runtime and verifies it. Any other project dependency conflict still fails.

### Initial Installation Cell

```python
{install}
```

Keep Colab's GPU-matched scientific stack. The imports cell checks the project's dependency tree;
an unrelated preinstalled package warning is not evidence that this project works or fails.
Do not suppress a real dependency conflict or replace TensorFlow/NumPy pins indiscriminately.

## Execution Order

Both columns below are one-based. "Notebook position" counts markdown cells too; Colab sections
are the easiest way to identify cells. Run top-to-bottom, not just the numbered training sections.

{execution}

The feature-cache cell processes the full corpora. Feature ablation and model comparison require
real training in Colab. Do not run the locked-test section until validation choices are frozen.
Saving precedes reload-tested inference, which precedes Gradio launch. GitHub documentation CI
does not run any of those expensive steps.

## Configuration

{configuration}

## Datasets And Splits

{corpora}

Both sexes and all valid CREMA-D intensity levels are retained. Calm and surprise are excluded
to harmonize the six labels. The manifest records path, dataset, dataset-prefixed speaker ID,
sex, emotion, source ID, and split. Duplicate source recordings from mirrors are removed.
Expected speaker counts are RAVDESS 24, CREMA-D 91, TESS 2, and SAVEE 4.

Original recordings are split before augmentation or feature extraction. Larger corpora target
70/15/15 speaker partitions. SAVEE assigns 2/1/1 speakers. TESS assigns one speaker to training,
one to test, and none to validation; this is not an exact 70/15/15 clip split. Assertions check
speaker/source separation, label coverage, deterministic membership, and train-only augmentation.
Dataset-specific test reports reveal corpus bias but are not cross-corpus generalization tests.

## Audio And Features

The shared preprocessing function validates input, converts stereo to mono, resamples, removes
DC offset, trims leading/trailing silence, safely peak-normalizes, and center-crops or zero-pads.
This is standardization, not a denoising model. Empty, silent, corrupt, non-finite, unsupported,
or excessively long inputs raise errors. Training and inference use the same saved configuration.

Training keeps the original plus one seeded augmented version with at most two transformations:
noise at 15-30 dB SNR, stretch 0.90-1.10, pitch shift within 1.5 semitones, or gain within 3 dB.
Validation and test have no augmentation. No circular boundary-wrapping time shift is used.

Default recurrent input is `({frames}, selected_channels)`, batched as
`(samples, time_frames, feature_channels)`. Time, not feature type, is the sequence axis.

{feature_rows}

MFCCs describe the spectral envelope; delta and delta-delta coefficients capture local movement
and acceleration. RMS describes the energy contour; ZCR helps distinguish voicing/noisiness.
Centroid, bandwidth, and rolloff describe brightness and spectral distribution. The LSTM
validation ablation keeps the smallest feature candidate within 0.005 macro F1 of the best.
This tolerance is a practical selection rule, not evidence of statistical equivalence.
StandardScaler is fitted only on training frames and persisted with the selected model.
Feature-cache fingerprints include configuration, manifest membership, file metadata, and versions.

## Models And Selection

Compared model keys: {', '.join(f'`{name}`' for name in models)}.
The current model builder below is extracted from the notebook, so architecture edits appear here:

```python
{model_code}
```

All architectures share splits, selected features, class weights from training labels, batch size,
seed, optimizer settings, maximum epochs, validation-loss early stopping, and learning-rate reduction.
Checkpoint selection uses validation loss; architecture selection uses validation macro F1 with
validation accuracy as a tie-breaker. Fixed seeds improve repeatability, but not necessarily
bitwise equality across different GPU hardware or software versions. No architecture is called
better until a completed experiment supports that claim.

## Evaluation And Artifacts

Evaluation reports accuracy, macro/weighted precision, recall, and F1; per-class support and
performance; raw and row-normalized confusion matrices; per-corpus metrics; loss/accuracy curves;
misclassified audio examples; and a reliability diagram with expected calibration error.
Confidence is a softmax probability, not a calibrated claim about a person's mental state.

Under `{config['project_dir']}/artifacts`, the notebook saves:

- `best_model.keras`, `feature_scaler.joblib`, `metadata.json`: the inference bundle.
- `manifest.csv`: exact source/speaker/split membership; review paths before publishing.
- `feature_ablation.csv`, `validation_comparison.csv`, `model_comparison.csv`: real experiment tables.
- Model-specific `.keras` checkpoints and `_history.json` files.
- `test_classification_report.csv`, `test_per_dataset.csv`, `calibration.json`.
- `confusion_matrices.png`, `training_curves.png`, `reliability_diagram.png`.

## Inference And Demo

After the save/reload section, call `predict_emotion('/path/to/sample.wav')` or pass
`(sample_rate, waveform)`. The predictor loads artifacts once, restores saved preprocessing,
checks model/scaler/label dimensions, and returns `emotion`, `confidence`, and all class
`probabilities`. The notebook compares saved/reloaded probabilities with the in-memory model.
Only load `.keras` and especially joblib artifacts you trust.

Gradio accepts upload or microphone input, provides audio playback, and displays the label,
confidence, and six probabilities. Launch uses a temporary public share link; do not submit
private recordings without consent. Microphone access may require browser permission.
To serve a previously trained bundle in a fresh runtime, first run setup/imports and define
the shared preprocessing/feature functions; do not rerun their dataset-dependent example calls.
The supported first-run path is the complete top-to-bottom Colab notebook.

## Troubleshooting

- Authentication: enable notebook access for both Colab Secrets; verify the legacy Kaggle key.
- Download: check dataset access, Kaggle mirror availability, runtime disk space, and parsed speaker counts.
- Dependency error: start a fresh runtime, run setup, restart if requested, then rerun imports.
- No GPU: change runtime hardware, then reconnect and rerun setup/imports.
- RAM/disk pressure: inspect the full-corpus feature cache and use a higher-RAM runtime where available.
- Missing model/scaler: finish training, evaluation, and saving before inference or Gradio.
- Stale documentation: follow [maintenance instructions](MAINTENANCE.md).

## Interview And Responsible Use

"I fixed leakage before increasing model complexity. I split by speaker, augmented only training
data, and fitted scaling only on training frames. Time-major acoustic features let recurrent
layers model prosodic changes. A validation ablation justified the features, then three models
were compared under shared rules. Selection was frozen before test evaluation, and model,
scaler, labels, and preprocessing were saved together so the demo used the training pipeline."

These acted English datasets have speaker, microphone, script, and corpus biases. TESS has just
two actresses. Results are not evidence of real-world mental-state detection and must not be
used for clinical, hiring, surveillance, or truth-detection decisions. Check each dataset's
license, including non-commercial restrictions, before redistribution or commercial use.

## Dependency Ranges

These are project compatibility ranges, not a frozen export of an entire Colab runtime.
The installation cell above intentionally installs only the necessary app/audio additions.

```text
{requirements}
```
"""
    status = ("The following values are read directly from committed `results/model_comparison.csv`.\n"
              "The documentation builder validates schema and numeric ranges, not experimental provenance.\n"
              "The maintainer must publish only outputs of completed Colab runs, with their configuration."
              if has_results else "No completed experiment table has been published. All metrics remain **Run required**.\n"
              "Pre-training smoke checks are not training or evaluation results.")
    results = f"""# Experiment Results

<!-- Generated by generate_documentation.py. Do not manually fill in metrics here. -->

{status}

{table}

{'Source: [model_comparison.csv](../results/model_comparison.csv).' if has_results else 'Run the ablation, comparison, and locked evaluation sections in Colab to obtain real values.'}

Model selection uses validation macro F1, not these test scores. The historical 78.57% result is
not comparable under the corrected speaker-disjoint protocol. No model superiority is claimed.
See [publishing instructions](MAINTENANCE.md#publishing-real-results).
"""
    for name in ("confusion_matrices.png", "training_curves.png", "reliability_diagram.png"):
        if (root / "results" / name).is_file():
            results += f"\n## {name.removesuffix('.png').replace('_', ' ').title()}\n\n![{name}](../results/{name})\n"
    readme = (root / "README.md").read_text(encoding="utf-8")
    return {"docs/PROJECT_GUIDE.md": guide, "docs/RESULTS.md": results,
            "README.md": replace_results(readme, table)}


def generate(root=ROOT, check=False):
    outputs = render(Path(root))
    stale = []
    for name, content in outputs.items():
        path = Path(root) / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            stale.append(name)
            if not check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8", newline="\n")
    if check and stale:
        raise ValueError("Outdated documentation: " + ", ".join(stale) + ". Run python generate_documentation.py")
    return stale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on outdated documentation without writing files")
    args = parser.parse_args()
    try:
        changed = generate(check=args.check)
    except (ValueError, KeyError, StopIteration, SyntaxError) as error:
        parser.exit(1, f"Documentation error: {error}\n")
    print("Documentation is up to date." if not changed else "Updated: " + ", ".join(changed))
