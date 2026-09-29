# Documentation Maintenance

## What Updates Automatically

`docs/PROJECT_GUIDE.md`, `docs/RESULTS.md`, and the marked results table in `README.md` are generated.
The guide reads notebook configuration, dataset handles, feature widths, model code, execution
order, and dependency ranges directly from the source files. Results stay unfilled until a real
comparison CSV is published. The rest of README and this maintenance page are maintained by hand.

Every branch push runs `.github/workflows/update-documentation.yml`. It checks the documentation
generator, refreshes these three files, and commits only changed generated documentation back
to the same branch. No model training, corpus download, Kaggle Secrets, or external token is needed.
Pull requests run read-only freshness checks; fork pull requests cannot write to this repository.
The workflow can also be started from **Actions > Maintain Documentation > Run workflow**.

Locally, the existing notebook rebuild now refreshes documentation too:

```bash
python build_professional_notebook.py
python generate_documentation.py --check
```

To refresh documentation without rebuilding or changing the notebook:

```bash
python generate_documentation.py
python -m unittest discover -s tests -p "test_documentation.py" -v
```

Both documentation generation and its tests require only Python 3.10+ standard-library modules.
They do not import TensorFlow, authenticate Kaggle, or execute notebook cells.

## Editing Rules

- Prefer changing `build_professional_notebook.py` and rebuilding the professional notebook.
- If you edit the notebook in Colab, synchronize those source edits into the Python generator
  before rebuilding. The guide warns if their cell sources differ. Documentation CI never
  overwrites a notebook or removes its outputs.
- Do not edit the generated guide, results page, or README's generated table by hand.
- Update explanatory prose in README and this page when behavior or project scope changes.
- Never commit Kaggle keys, access tokens, `.env` files, `kaggle.json`, raw audio, or caches.
- Historical notebooks are preserved locally but excluded from this professional publication;
  one contained an exposed token. Revoke that token with its provider.

GitHub records each change in commit history. Generated files use a source fingerprint instead
of a wall-clock timestamp, so re-running unchanged sources produces no noisy documentation commits.
Automatic updates apply to changes pushed to GitHub, not unsaved edits in a live Colab session.

## Publishing Real Results

1. Complete feature ablation, model comparison, and the locked test evaluation in Colab.
2. Download the artifact directory before the runtime is deleted.
3. Publish `model_comparison.csv` under `results/` without manually substituting scores.
   Values must be fractions in `[0, 1]`, not percentages, and contain all three compared models.
4. For visual results, publish the generated `confusion_matrices.png`, `training_curves.png`,
   and `reliability_diagram.png` directly under `results/`. The results page links present figures.
5. Also retain configuration/label metadata and `feature_ablation.csv`, `validation_comparison.csv`,
   `test_classification_report.csv`, `test_per_dataset.csv`, and `calibration.json` for traceability.
   Review paths and other sensitive information before publishing metadata or manifests.
6. Push the files. GitHub refreshes the results table and page. Check the workflow finishes successfully.

Only completed, real runs may be published. Numeric validation cannot establish that an experiment
was run correctly. Do not select models based on test results or recycle the compromised historical score.
Changing an experiment's sources does not retrain existing artifacts: re-run in Colab and publish
the matching configuration when replacement results are available.

## Workflow Permissions And Failures

The update job requests only `contents: write`; the pull-request job has `contents: read`.
GitHub's built-in `GITHUB_TOKEN` is enough for an ordinary unprotected branch. Its push does not
trigger another push workflow, preventing a documentation loop.
See [GitHub's trigger rules](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow).

If Actions is disabled or write permission is blocked by an organization policy, enable the allowed
Actions/workflow permissions in repository settings. If branch protection requires pull requests,
run the generator locally and include the documentation changes in your pull request; do not
disable branch protection just to let a documentation bot push. An out-of-date branch run exits
without force-pushing; the newer push run handles the latest changes.

Invalid/incomplete result CSVs fail the job instead of silently printing invented values. Open
the failed Actions log, correct the source data or generator, regenerate, and push the correction.
