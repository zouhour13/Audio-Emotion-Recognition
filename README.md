# Speech Emotion Recognition

An end-to-end, leakage-aware speech emotion recognition project for six acted emotions: **angry, disgust, fear, happy, neutral, and sad**. The project combines RAVDESS, CREMA-D, TESS, and SAVEE, compares three sequence models, saves a reproducible inference bundle, and serves predictions through Gradio.

> The `78.57%` score in the historical notebook is not a valid held-out benchmark. It used the same split for validation and testing, mixed speakers across splits, augmented evaluation audio, transposed the sequence axes, and displayed an incorrect label map. Run the professional notebook to produce replacement results.

## Documentation

See the [project guide](docs/PROJECT_GUIDE.md) for configuration, exact Colab cell order, datasets,
feature/model architecture, evaluation, inference, and troubleshooting. The [results page](docs/RESULTS.md)
contains published experiment outputs only. [Maintenance instructions](docs/MAINTENANCE.md) explain how
documentation is refreshed automatically on every GitHub branch push and on local notebook rebuilds.
Live Colab edits/results must be published to the repository before GitHub can document them.

[Open the professional notebook in Colab](https://colab.research.google.com/github/zouhour13/Audio-Emotion-Recognition/blob/main/ser_professional_colab.ipynb).

## Pipeline

`Raw WAV -> metadata/validation -> speaker-disjoint split -> mono/16 kHz -> silence trim -> normalization -> 4-second crop/pad -> train-only augmentation -> time-major acoustic features -> validation feature ablation -> model comparison -> locked test evaluation -> saved inference bundle -> Gradio`

## What Makes The Evaluation Credible

- Original files are split by speaker before augmentation or feature extraction.
- Validation chooses features and architecture; the test set is evaluated only after choices are frozen.
- Feature standardization is fitted on training frames only.
- All models share the same data, selected features, class weights, seed, and training policy.
- Overall, per-class, and per-dataset results expose class and corpus bias.

## Run In Colab

1. Open [`ser_professional_colab.ipynb`](ser_professional_colab.ipynb).
2. Select a GPU runtime.
3. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to Colab Secrets and enable notebook access for both secrets. Use the username and API key from Kaggle Settings, not your email or password.
4. Run the first code cell (package setup), restart the session if Colab requests it, and run the next code cell (imports, dependency checks, and GPU detection). Run the remaining cells in order. Dataset downloads, features, models, plots, and metadata are stored under `/content/ser_project`.
5. Download the generated `artifacts/` directory before the Colab runtime is deleted. The Gradio cell creates a temporary public share link.

The setup cell deliberately does not reinstall TensorFlow, NumPy, pandas, scikit-learn, or Matplotlib. Colab supplies a mutually compatible GPU stack; forcing historical pins can produce `ResolutionImpossible` or disconnect TensorFlow from the runtime CUDA libraries. The notebook installs Gradio 6 to match Colab's current Hugging Face and Starlette dependency family.

The notebook can also use already-extracted dataset folders by changing `DATA_ROOTS` in the configuration cell.

The imports cell checks the project's dependency tree and reports real mismatches. A global `pip check` can additionally report conflicts among unrelated packages that Colab preinstalls; those should be assessed rather than interpreted as proof that all project imports work. The setup includes `jedi`, which some Colab IPython environments need.

Corpus validation requires all four datasets and the expected speaker counts. It removes duplicate source recordings in mirrors and saves the exact split manifest. Cached features are keyed by the manifest, file metadata, seed, audio configuration, and feature-library versions; interrupted cache writes are not reused. Audio preprocessing and model forward-pass checks run before their respective expensive stages.

For local checks without dataset downloads or training, run `python verify_professional_notebook.py` in an environment with NumPy, pandas, librosa, soundfile, scikit-learn, Gradio, and nbformat. Add `nbformat>=5.10,<6` only to that verification environment. These checks do not establish model performance or CUDA compatibility.

Pre-training verification on September 29, 2026 passed in the live Colab GPU session: TensorFlow 2.20.0, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, librosa 0.11.0, Gradio 6.28.0, and Kaggle 1.8.4. The project dependency check passed; each architecture completed one tiny synthetic GPU batch and a native `.keras` save/reload check. Colab Secrets and authenticated metadata access worked for all four Kaggle handles. Local audio, parser, split, cache, scaler, and Gradio construction checks also passed. Full corpus downloads, feature ablation, real training, and final evaluation remain to be run in Colab.

## Experiments

The notebook first performs validation-only feature ablation, then compares LSTM, BiLSTM, and CNN-BiLSTM. This table is populated only from a real `results/model_comparison.csv` published after Colab evaluation; until then it stays unfilled.

<!-- BEGIN GENERATED RESULTS -->
| Model | Parameters | Best epoch | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LSTM | Run required | Run required | Run required | Run required | Run required | Run required | Run required |
| BiLSTM | Run required | Run required | Run required | Run required | Run required | Run required | Run required |
| CNN-BiLSTM | Run required | Run required | Run required | Run required | Run required | Run required | Run required |
<!-- END GENERATED RESULTS -->

## Artifacts And Inference

Training writes the selected model as `best_model.keras`, the training-only scaler as `feature_scaler.joblib`, and label/preprocessing/feature metadata as JSON. Inference loads the saved configuration and checks the model/scaler dimensions and label order. `predict_emotion()` returns the predicted emotion, confidence, and all six probabilities. The Gradio app accepts uploaded files and microphone recordings, supports playback, and shows clear input errors for empty, silent, invalid, unsupported, or longer-than-60-second recordings.

The bundle also includes the split manifest, validation/model comparisons, training histories, per-class and per-dataset reports, calibration statistics, and PNG evaluation figures. The notebook verifies that predictions match after saving and reloading. Evaluation values remain unfilled until training and evaluation run in Colab.

## Interview Summary

The central design decision is speaker-disjoint evaluation. A random clip split lets a model recognize voices and recording conditions instead of emotion. Audio is standardized to a fixed waveform, but augmentation is limited to realistic, seeded transformations and is applied only to training data. Features are stored as `(time frames, feature channels)`, allowing recurrent layers to model how prosody evolves over time. MFCC deltas capture movement, RMS captures energy, and ZCR captures voicing/noisiness; extra spectral features are retained only if validation macro F1 justifies them. Model selection uses validation macro F1 because it gives each emotion equal importance, while the untouched test set is reserved for one final comparison.

## Limitations And Responsible Use

These are acted English datasets recorded under different laboratory conditions. TESS has only two speakers and can dominate corpus cues. Predictions should not be interpreted as a clinical, psychological, hiring, surveillance, or truth-detection assessment. Review each source dataset's license before redistribution or commercial use.
