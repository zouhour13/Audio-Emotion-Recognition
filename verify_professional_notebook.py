"""Check notebook contracts without corpus downloads or neural-network training."""

import ast
import gc
import hashlib
import json
import os
import re
import sys
import tempfile
import types
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import joblib
import librosa
import nbformat
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent


def load_definitions(notebook, namespace):
    names = {"RAV_MAP", "CREMA_MAP", "TESS_MAP", "TESS_SPEAKERS", "SAVEE_MAP", "FEATURE_SETS"}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = "\n".join(line for line in cell.source.splitlines() if not line.startswith("%"))
        tree = ast.parse(source)
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]
        for node in nodes:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<notebook definitions>", "exec"), namespace)
            if isinstance(node, ast.ClassDef) and node.name == "Config":
                namespace["CFG"] = namespace["Config"]()


def expect_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("Expected a ValueError")


def check_kaggle(namespace):
    calls = []

    class DummyAPI:
        def authenticate(self):
            assert os.environ["KAGGLE_USERNAME"] == "fixture-user"
            assert os.environ["KAGGLE_KEY"] == "fixture-key"

    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: calls.append(name) or {
        "KAGGLE_USERNAME": " fixture-user ", "KAGGLE_KEY": " fixture-key "}[name])
    google = types.ModuleType("google")
    google.colab = colab
    kaggle = types.ModuleType("kaggle")
    api_package = types.ModuleType("kaggle.api")
    extended = types.ModuleType("kaggle.api.kaggle_api_extended")
    extended.KaggleApi = DummyAPI
    modules = {"google": google, "google.colab": colab, "kaggle": kaggle,
               "kaggle.api": api_package, "kaggle.api.kaggle_api_extended": extended}
    with patch.dict(sys.modules, modules), patch.dict(os.environ, {}, clear=False):
        namespace["configure_kaggle"]()
        assert calls == ["KAGGLE_USERNAME", "KAGGLE_KEY"]
        colab.userdata.get = lambda _: (_ for _ in ()).throw(RuntimeError("Access denied"))
        try:
            namespace["configure_kaggle"]()
        except RuntimeError as exc:
            assert "enable notebook access" in str(exc)
        else:
            raise AssertionError("Denied secrets must not fall back to stale environment credentials")
    print("PASS: Colab Secrets authentication (mocked, no credentials transmitted)")


def check_manifest(namespace, root):
    root.mkdir()
    roots = {name: root / name for name in ("ravdess", "cremad", "tess", "savee")}
    for folder in roots.values():
        folder.mkdir()
    for actor in range(1, 25):
        for code in namespace["RAV_MAP"]:
            (roots["ravdess"] / f"03-01-{code}-01-01-01-{actor:02d}.wav").touch()
    duplicate = roots["ravdess"] / "mirror"
    duplicate.mkdir()
    (duplicate / "03-01-01-01-01-01-01.wav").touch()
    for actor in range(1001, 1092):
        for code in namespace["CREMA_MAP"]:
            (roots["cremad"] / f"{actor}_DFA_{code}_XX.wav").touch()
    for actor in ("OAF", "YAF"):
        for emotion in namespace["TESS_MAP"]:
            (roots["tess"] / f"{actor}_back_{emotion}.wav").touch()
    for emotion in namespace["TESS_MAP"]:
        (roots["tess"] / f"MIRROR_back_{emotion}.wav").touch()
    for actor in ("DC", "JE", "JK", "KL"):
        for code in namespace["SAVEE_MAP"]:
            (roots["savee"] / f"{actor}_{code}01.wav").touch()
    manifest = namespace["prepare_dataset"](roots)
    split = namespace["assign_speaker_splits"](manifest)
    assert len(manifest) == (24 + 91 + 2 + 4) * 6
    assert split.equals(namespace["assign_speaker_splits"](manifest))
    assert split.groupby("speaker_id").split.nunique().max() == 1
    assert split.groupby("source_id").split.nunique().max() == 1
    assert set(split.loc[split.dataset == "tess", "split"]) == {"train", "test"}
    assert set(namespace["TESS_SPEAKERS"]) == {"OAF", "YAF"}
    assert not any(row["speaker_id"] == "tess:MIRROR" for row in namespace["parse_tess"](roots["tess"]))
    savee_folder = roots["savee"] / "DC"
    savee_folder.mkdir()
    (savee_folder / "sa02.wav").touch()
    assert any(r["speaker_id"] == "savee:DC" and r["emotion"] == "sad"
               for r in namespace["parse_savee"](roots["savee"]))
    print("PASS: all corpus parsers, mirror deduplication, deterministic speaker splits")


def check_audio_and_cache(namespace, root):
    cfg = namespace["CFG"]
    load, preprocess, extract = (namespace[name] for name in ("load_audio_file", "preprocess_audio", "extract_features"))
    tone = (.2 * np.sin(2 * np.pi * 220 * np.arange(4000) / 8000)).astype(np.float32)
    for wave in (tone, np.stack([tone, tone]), np.column_stack([tone, tone])):
        clean, _ = preprocess(wave, 8000)
        assert clean.shape == (64000,) and clean.dtype == np.float32
        assert extract(clean).shape == (401, 44)
    long_tone = np.tile(tone, 14)
    assert preprocess(long_tone, 8000)[0].shape == (64000,)
    for invalid in (np.array([]), np.zeros(800), np.array([np.nan]), np.ones((2, 3, 4))):
        expect_error(preprocess, invalid, 16000)
    expect_error(preprocess, tone, 0)
    expect_error(preprocess, np.tile(tone, 122), 8000)
    corrupt = root / "corrupt.wav"
    corrupt.write_bytes(b"not a wave file")
    expect_error(load, corrupt)
    expect_error(load, root / "missing.wav")
    unsupported = root / "unsupported.txt"
    unsupported.touch()
    expect_error(load, unsupported)
    too_long = root / "too_long.wav"
    sf.write(too_long, np.tile(tone, 122), 8000)
    expect_error(load, too_long)
    first = namespace["augment_audio"](clean, 16000, np.random.default_rng(42))
    second = namespace["augment_audio"](clean, 16000, np.random.default_rng(42))
    assert np.array_equal(first, second) and first.shape == clean.shape
    assert np.isfinite(first).all()
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        path = root / f"audio_{index}.wav"
        sf.write(path, tone, 8000)
        row = namespace["record"](path, "fixture", str(index), "unknown", namespace["LABELS"][index])
        rows.append({**row, "split": split})
    manifest = pd.DataFrame(rows)
    arrays = namespace["build_feature_cache"](manifest)
    assert arrays[0].shape == (4, 401, 44)
    assert arrays[5].sum() == 1 and not arrays[5][arrays[2] != "train"].any()
    with patch.dict(namespace, {"load_audio_file": lambda _: (_ for _ in ()).throw(AssertionError("Cache not reused"))}):
        cached = namespace["build_feature_cache"](manifest)
        assert all(np.array_equal(a, b) for a, b in zip(arrays, cached))
    modified = manifest.copy()
    modified.loc[1, "split"] = "train"
    assert namespace["build_feature_cache"](modified)[0].shape[0] == 5
    assert len(list(namespace["CACHE_DIR"].glob("expanded_features_*.npz"))) == 2
    print("PASS: mono/stereo/resampling, invalid audio, seeded augmentation, temporal shape and cache invalidation")


def check_scaler_and_inference(namespace, root):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(4, 401, 44)).astype(np.float32)
    features[2:] += 100
    namespace.update(X_all=features, splits_all=np.array(["train", "train", "validation", "test"]))
    scaled, scaler = namespace["scaled_views"]("core")
    assert np.allclose(scaler.mean_, features[:2, :, :41].mean(axis=(0, 1)), atol=1e-6)
    assert abs(scaled[:2].mean()) < 1e-5 and scaled[2:].mean() > 10
    cfg = replace(namespace["CFG"], sample_rate=8000, duration=1.0, n_mfcc=5)
    columns = list(range(15))
    tone = (.2 * np.sin(2 * np.pi * 220 * np.arange(4000) / 8000)).astype(np.float32)
    clean, _ = namespace["preprocess_audio"](tone, 8000, cfg)
    feature = namespace["extract_features"](clean, sr=8000, config=cfg)[:, columns]
    scaler = StandardScaler().fit(feature)
    root.mkdir()
    joblib.dump(scaler, root / "feature_scaler.joblib")
    meta = {"labels": namespace["LABELS"], "label_to_id": namespace["LABEL_TO_ID"],
            "config": asdict(cfg), "feature_columns": columns, "input_shape": list(feature.shape)}
    (root / "metadata.json").write_text(json.dumps(meta))

    class FakeModel:
        input_shape = (None, *feature.shape)
        output_shape = (None, 6)

        def predict(self, batch, verbose=0):
            self.last_input = batch
            return np.full((len(batch), 6), 1 / 6, dtype=np.float32)

    model = FakeModel()
    namespace["keras"] = types.SimpleNamespace(models=types.SimpleNamespace(load_model=lambda *a, **k: model))
    predictor = namespace["EmotionPredictor"](root)
    result = predictor.predict_emotion((8000, tone))
    assert set(result["probabilities"]) == set(namespace["LABELS"])
    assert np.isclose(sum(result["probabilities"].values()), 1)
    assert np.allclose(model.last_input[0], scaler.transform(feature))
    expect_error(predictor.predict_emotion, None)
    expect_error(predictor.predict_emotion, (8000, np.zeros(20)))
    expect_error(predictor.predict_emotion, (8000, tone, "extra"))
    namespace["predictor"] = predictor
    print("PASS: training-only scaler, artifact config/label checks, inference contract (mock neural model)")


def main():
    notebook = nbformat.read(ROOT / "ser_professional_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        cache = root / "cache"
        cache.mkdir()
        namespace = {"__name__": "__main__", "CFG": types.SimpleNamespace(seed=42, sample_rate=16000),
                     "LABELS": ["angry", "disgust", "fear", "happy", "neutral", "sad"],
                     "DATA_ROOTS": {}, "ARTIFACT_DIR": root, "CACHE_DIR": cache}
        namespace.update({name: globals()[name] for name in
                          ("dataclass", "asdict", "Path", "np", "pd", "librosa", "sf", "os", "re", "json", "hashlib",
                           "joblib", "StandardScaler", "version", "tqdm", "accuracy_score", "precision_recall_fscore_support", "gc")})
        namespace["LABEL_TO_ID"] = {label: i for i, label in enumerate(namespace["LABELS"])}
        load_definitions(notebook, namespace)
        namespace["CFG"] = namespace["Config"](project_dir=str(root))
        check_kaggle(namespace)
        check_manifest(namespace, root / "datasets")
        check_audio_and_cache(namespace, root)
        check_scaler_and_inference(namespace, root / "bundle")
        demo_cell = next(c for c in notebook.cells if c.cell_type == "code" and "with gr.Blocks" in c.source)
        tree = ast.parse(demo_cell.source)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Expr)
                     and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                     and node.value.func.attr == "launch")]
        exec(compile(tree, "<Gradio UI construction>", "exec"), namespace)
        assert isinstance(namespace["demo"], gr.Blocks)
        namespace["demo"].close()
        print("PASS: Gradio 6 interface construction and callback wiring (no server launched)")
    print("Notebook schema and every code cell compile. No corpus downloads or real model training were run.")


if __name__ == "__main__":
    main()
