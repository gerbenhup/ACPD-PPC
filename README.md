# Anomalous Change Point Detection Using Probabilistic Predictive Coding

Reference implementation for the paper "Anomalous Change Point Detection Using Probabilistic
Predictive Coding". `PPC.py` contains the shared framework (model wrappers, training loops,
losses, probability/anomaly-scoring utilities); `PPC_examples/` contains the three experiments
from the paper.

## Setup

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

## License

This project is licensed under the [MIT License](LICENSE).