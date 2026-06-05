User Guide
=====================

Both ``SomClassifier`` and ``SomVQ`` implement the scikit-learn API and can be used as drop-in replacements for other scikit-learn estimators, including full compatibility with ``Pipeline``, ``GridSearchCV``, and ``cross_val_score``.

Key Parameters
--------------

Both estimators share the following most important parameters:

- ``spreading_factor`` (default 0.5) — controls the growing threshold. Higher values produce more neurons and finer resolution; lower values produce fewer neurons.
- ``max_neurons`` (default 100) — hard upper limit on the number of neurons.
- ``n_iter`` (default 500) — maximum number of training epochs.
- ``metric`` (default ``"euclidean"``) — distance metric; ``"cosine"`` is also supported.
- ``pointer_search`` (default ``"fine"``) — accelerates BMU search by restricting winner
  lookup to the previous winner's graph neighbourhood. ``"fine"`` gives ~3× speedup in the
  fine phase with near-identical quality. ``"all"`` extends pointer search to the coarse
  phase (faster, slightly lower quantization accuracy, improved topographic error).
  ``pointer_search_radius`` (default 1) controls the search radius in graph hops.
- ``winner_stability_threshold`` (default 0.01) — convergence criterion for the coarse phase:
  training is considered converged when fewer than this fraction of samples change their BMU
  between epochs. Set to ``None`` to use weight-delta convergence instead.

Classification
--------------

.. code-block:: python

    from dbgsom import SomClassifier
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = SomClassifier(spreading_factor=0.5, max_neurons=80)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

.. code-block:: pycon

    >>> 0.9333...

.. code-block:: python

    clf.predict(X_test)

.. code-block:: pycon

    >>> array([0, 1, 8, ..., 8, 9, 6])

.. code-block:: python

    clf.predict_proba(X_test)   # class probability per sample

Clustering / Vector Quantization
---------------------------------

.. code-block:: python

    from dbgsom import SomVQ
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)

    som = SomVQ(spreading_factor=0.5, max_neurons=80)
    labels = som.fit_predict(X)   # fit and assign cluster labels in one step

.. code-block:: python

    som.quantization_error_   # average distance from samples to their prototype
    som.topographic_error_    # fraction of samples with topographic errors
    som.n_iter_               # number of epochs actually used

Transform
---------

Both estimators implement ``transform()``, which represents each sample as a sparse non-negative linear combination of the prototype weight vectors. This yields an ``(n_samples, n_prototypes)`` coefficient matrix useful for downstream tasks.

.. code-block:: python

    coefs = som.transform(X)   # shape (n_samples, n_prototypes)

Reference: Teuvo Kohonen, *Description of Input Patterns by Linear Mixtures of SOM Models*, 2007.

scikit-learn Integration
------------------------

Because both estimators follow the scikit-learn API, they work with standard tools:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("som", SomVQ(spreading_factor=0.5, max_neurons=80)),
    ])
    pipe.fit(X)
