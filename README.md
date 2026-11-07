# Anomalous Change Point Detection Using Probabilistic Predictive Coding

Reference implementation for the paper "Anomalous Change Point Detection Using Probabilistic
Predictive Coding". The `PPC` package contains the shared framework (model wrappers, training
loops, losses, probability/anomaly-scoring utilities), split into one module per concern:

| Module | Contents |
| --- | --- |
| `PPC.trainers` | `PPC` and `PCPC` trainer classes: train/train_step/test_step |
| `PPC.anomaly` (`PPC.Anomaly`) | Online anomaly-scoring class |
| `PPC.losses` | `MeanAbsoluteError`, `MeanSquaredError`, `MaximumLogLikelihoodLoss` (Gaussian or Laplace prior), `ContrastiveNLLRLoss` (PCPC) |
| `PPC.probability` | Gaussian (log-)likelihoods, Laplace log-likelihoods, Mahalanobis distance, tail probabilities |
| `PPC.dimension` | `PCA` wrapper (with Gaussian projection), `TSNE` wrapper |
| `PPC.wrappers` | `TimeDistributed`, `TimeStacked` Keras model wrappers |
| `PPC.weights` | Loss time-weighting schemes |
| `PPC.math` | Numerically-safe `log`/`exp`/`square`/`sqrt`/`abs` |
| `PPC.callbacks` | `TerminateOnNaN`, `RestoreOnEarlyStopping`, `EarlyStopping`, `ReduceLROnPlateau` |
| `PPC.tools` | `Settings`, `Arguments`, `num_trainable_weights`, PPC/PCPC model save/load helpers |
| `PPC.gpu` | GPU discovery/selection via `nvidia-smi` |

`PPC_examples/` contains the three experiments from the paper. `PCPC_examples/` contains the
Probabilistic Contrastive Predictive Coding (PCPC) variant on two of these experiments.

### Laplace prior

By default the forecast models predict a Gaussian distribution over the future encodings (mean and
standard deviation per encoding dimension). To use a Laplace distribution instead (location and
scale), pass `distribution='Laplace'` to the forecast loss:

```python
PPC.losses.MaximumLogLikelihoodLoss(PPC.weights.Uniform(Nf), warmup=False, distribution='Laplace')
```

The forecast model architecture stays the same; only the interpretation of its two output channels
changes.

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

## PCPC experiments

`PCPC_examples/` trains the encoder, autoregressive and forecast models without a decoder. Instead
of a reconstruction loss, the forecast distribution is trained contrastively
(`PPC.losses.ContrastiveNLLRLoss`): the likelihood of the true future sample is maximized relative
to the mean likelihood of `Nchi` negative samples. Models are trained by `PPC.trainers.PCPC` and
saved to `models/<hash>/`. As above, run the scripts from within each example's directory.

The PCPC examples use the same settings as their PPC counterparts, so the results can be compared
directly:

- Same encoder, autoregressive and forecast models (PCPC has no decoder). The only difference is
  that the PCPC forecast models bound the predicted standard deviation from below, because the
  contrastive loss uses the likelihood itself, which would otherwise overflow in float32.
- Same training procedure: RMSprop (learning rate 1e-4), 1000 warm-up iterations with unit
  standard deviation, then training until the validation forecast loss has not improved for 10000
  iterations, restoring the best weights.
- Same data: training and validation samples from the same distributions and splits (PCPC adds 9
  negative samples to each), identical test sets, and the same `test.py` (same figures and
  metrics).

### `PCPC_examples/sine_frequency`

Datasets must be generated once before training (~60 GB of `.h5` files): training and validation
sets of 1,000,000 and 100,000 samples with 9 negative samples each, and the test sets. The test sets
use the same settings and seeds as the PPC sine experiment, so they are identical to the PPC test
sets.

```
python generate_datasets.py
python train.py
python test.py
```

`test.py` produces `sine-examples.pdf`, `sine-matrix.pdf` and `sine-curves.pdf` in the model
directory, and prints the same detection metrics as the PPC sine experiment.

### `PCPC_examples/mnist_sequence`

MNIST is downloaded automatically by Keras on first use; negative samples are digits other than
the successor digit.

```
python train.py
python test.py
```

`test.py` produces `mnist-matrix.pdf`, `mnist-curves.pdf` and `mnist-latent-hexbin.pdf` in the
model directory, and prints the same detection metrics as the PPC MNIST experiment.

## Reproducibility

All randomness is seeded:

- Generated datasets: sample `i` of a dataset with seed `s` is generated with seed `(s, i)`, so the
  `.h5` files are identical on every run, regardless of the number of worker processes.
- Training: every dataset object owns its random number generator (`seed` argument), the global
  seeds are set with `tf.keras.utils.set_random_seed`, and
  `tf.config.experimental.enable_op_determinism()` makes TensorFlow operations deterministic, so
  rerunning `train.py` on the same hardware and software gives the same model. The exception is
  `PPC_examples/mnist_sequence`: the gradient of its `UpSampling2D` layers has no deterministic GPU
  implementation, so there the determinism switch is off and reruns on a GPU can differ slightly.
- Testing: the F1-optimal threshold search (`scipy.optimize.differential_evolution`) uses `seed=0`.

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