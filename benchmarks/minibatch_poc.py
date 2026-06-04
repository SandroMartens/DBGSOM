import time

from sklearn.datasets import fetch_openml, make_blobs
from sklearn.preprocessing import StandardScaler

from dbgsom.SomVQ import SomVQ

# ── Fashion-MNIST 10k (D=784) ────────────────────────────────────────────────
fmnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
X_fmnist = StandardScaler().fit_transform(fmnist.data[:10_000].astype(float))

PARAMS_FMNIST = dict(
    n_iter=500, spreading_factor=0.8, max_neurons=100, sigma_end=1, random_state=42
)
CONFIGS = [
    ("Batch   ", dict()),
    ("MB-4096 ", dict(batch_size=4096)),
    ("MB-2048 ", dict(batch_size=2048)),
    ("MB-1024 ", dict(batch_size=1024)),
]


def run_benchmark(X, params, configs, label):
    header = f"{'Config':<12} {'Time (s)':>8} {'QE':>8} {'TE':>8}"
    header += f" {'Neurons':>8} {'Epochs':>8}"
    print(f"\n{'─' * 60}")
    print(f"  {label}  (N={len(X):,}, D={X.shape[1]})")
    print(f"{'─' * 60}")
    print(header)
    print("-" * 60)
    for name, kwargs in configs:
        som = SomVQ(**params, **kwargs)
        t0 = time.perf_counter()
        som.fit(X)
        elapsed = time.perf_counter() - t0
        print(
            f"{name:<12} {elapsed:>8.1f} {som.quantization_error_:>8.4f}"
            f" {som.topographic_error_:>8.4f} {len(som.neurons_):>8} {som.n_iter_:>8}"
        )


# run_benchmark(X_fmnist, PARAMS_FMNIST, CONFIGS, "Fashion-MNIST 10k  D=784")

# ── Synthetic large-N, low-D ─────────────────────────────────────────────────
# N=200k, D=10: N×K distance matrix ≈ 200k×100×8 = 160MB → overflows L3
# B×K with B=4096:                  ≈ 4096×100×8 = 3.2MB  → L2-resident
N_SYNTH = 200_000
D_SYNTH = 10
X_synth, _ = make_blobs(
    n_samples=N_SYNTH, n_features=D_SYNTH, centers=30, random_state=42
)
X_synth = StandardScaler().fit_transform(X_synth)

PARAMS_SYNTH = dict(
    n_iter=200, spreading_factor=0.5, max_neurons=100, sigma_end=0.5, random_state=42
)
CONFIGS_SYNTH = [
    ("Batch   ", dict()),
    ("MB-16k  ", dict(batch_size=16_384)),
    ("MB-8k   ", dict(batch_size=8_192)),
    ("MB-4k   ", dict(batch_size=4_096)),
    ("MB-2k   ", dict(batch_size=2_048)),
]

run_benchmark(X_synth, PARAMS_SYNTH, CONFIGS_SYNTH, "Synthetic 200k  D=10")
