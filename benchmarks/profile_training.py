from line_profiler import LineProfiler
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from dbgsom.BaseSom import BaseSom
from dbgsom.SomVQ import SomVQ

fmnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
X = StandardScaler().fit_transform(fmnist.data[:10_000].astype(float))
y = fmnist.target[:10_000]

som = SomVQ(
    n_iter=2000,
    lambda_=15,
    max_neurons=300,
    sigma_end=1,
    verbose=True,
    random_state=42,
)

profiler = LineProfiler()
for method in [
    BaseSom._grow_som,
    BaseSom._get_winning_neurons,
    BaseSom._update_weights,
]:
    profiler.add_function(method)

profiler.runcall(som.fit, X)
profiler.print_stats(output_unit=1e-3)  # times in milliseconds
