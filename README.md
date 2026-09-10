# Anomalous Change Point Detection Using Probabilistic Predictive Coding

Reference implementation for the paper "Anomalous Change Point Detection Using Probabilistic
Predictive Coding". The `PPC` package contains the shared framework (model wrappers, training
loops, losses, probability/anomaly-scoring utilities), split into one module per concern:

| Module | Contents |
| --- | --- |
| `PPC.trainers` | `PPC` trainer class: train/train_step/test_step |
| `PPC.anomaly` (`PPC.Anomaly`) | Online anomaly-scoring class |
| `PPC.losses` | `MeanSquaredError`, `MaximumLogLikelihoodLoss` |
| `PPC.probability` | Gaussian log-likelihoods, Mahalanobis distance, tail probabilities |
| `PPC.dimension` | `PCA` wrapper (with Gaussian projection) |
| `PPC.wrappers` | `TimeDistributed`, `TimeStacked` Keras model wrappers |
| `PPC.weights` | Loss time-weighting schemes |
| `PPC.math` | Numerically-safe `log`/`exp`/`square` |
| `PPC.callbacks` | `TerminateOnNaN`, `RestoreOnEarlyStopping` |
| `PPC.tools` | `Settings`, model save/load helpers |
| `PPC.gpu` | GPU discovery/selection via `nvidia-smi` |

`PPC_examples/` contains the three experiments from the paper.

## Setup

Requires Python 3.9+ (for `list[int]`-style built-in generics) and a version compatible with
`tensorflow==2.10.0` (Python 3.7-3.10).

```
pip install -r requirements.txt
```

GPU selection is automatic (`PPC.gpu.find_available`): each script picks the first available GPU
with utilization below 80% and at least 5500 MiB free memory. This requires `nvidia-smi` to be on
`PATH`.

## Reproducing the experiments

Run each example's scripts from within its own directory (they resolve paths relative to
`__file__`), e.g. `cd PPC_examples/mnist_sequence`.

### `proportionality_test` (§4.1, "Proportionality test")

Data is generated on the fly during training, no separate generation step is needed.

```
python train.py
python test.py
```

`train.py` repeats training 100 times (matching the paper's N=100 sensitivity-to-initialization
analysis), seeding each repeat with its iteration index so individual runs are reproducible while
still varying across repeats. `test.py` evaluates every trained model, producing a `proptest.pdf`
per model (Figure 2 style) and printing the true-vs-fitted Gaussian parameter comparison
(Table 1), aggregated as mean ± std across all models.

### `sine_frequency` (§4.2, "Sine wave frequency deviation")

Datasets must be generated once before training (~17 GB of `.h5` files under
`PPC_examples/sine_frequency/`):

```
python generate_datasets.py
python train.py
python test.py
```

`generate_datasets.py` only generates datasets that don't already exist, so it's safe to rerun.
`test.py` evaluates every model directory under `models/`, producing `sine-examples.pdf`,
`sine-matrix.pdf`, `sine-curves.pdf`, and the full set of threshold-dependent detection metrics.

### `mnist_sequence` (§4.3, "Counting with MNIST digits")

MNIST is downloaded automatically by Keras on first use.

```
python train.py
python test.py
```

`train.py` trains one model and saves it to `models/<hash>/`, where `<hash>` is a deterministic
hash of the model's hyperparameters (`PPC.tools.Settings.__hash__`) — rerunning `train.py` with
the same settings resumes/skips training if the directory already exists. `test.py` evaluates
every model directory under `models/`, producing `mnist-matrix.pdf` (Figure 6), `mnist-curves.pdf`
(Figure 7), `mnist-latent-hexbin.pdf` (Figure 8), and printing the full set of threshold-dependent
detection metrics (best threshold selected on a validation split, evaluated on the test split).

## Citation

This code is archived on 4TU.ResearchData: [doi.org/10.4121/1943fd48-e26c-467e-af04-5f1349ca0bd0.v1](https://doi.org/10.4121/1943fd48-e26c-467e-af04-5f1349ca0bd0.v1).

```bibtex
@misc{hup2026ppc,
  author       = {Hup, Roelof and Merkofer, Julian and Bhogal, Alex and van Sloun, Ruud and Haakma, Reinder and Vullings, Rik},
  title        = {Code underlying the publication: Anomalous Change Point Detection Using Probabilistic Predictive Coding},
  year         = {2026},
  publisher    = {4TU.ResearchData},
  howpublished = {Software},
  doi          = {10.4121/1943fd48-e26c-467e-af04-5f1349ca0bd0.v1},
  url          = {https://doi.org/10.4121/1943fd48-e26c-467e-af04-5f1349ca0bd0.v1}
}
```

## License

This project is licensed under the [MIT License](LICENSE).