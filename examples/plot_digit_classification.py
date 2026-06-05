"""Digit Classification with SomClassifier
========================================

Classifies handwritten digits from the sklearn digits dataset using
``SomClassifier`` wrapped in a scikit-learn ``Pipeline`` with ``StandardScaler``.

The dataset contains 1 797 grayscale images (8×8 pixels, 64 features) of digits 0–9.
"""

# %%
# Build and Train Pipeline
# ------------------------
# ``SomClassifier`` is a drop-in replacement for any scikit-learn classifier.
# Wrapping it in a ``Pipeline`` applies ``StandardScaler`` automatically
# during both ``fit`` and ``predict``.

from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dbgsom.SomClassifier import SomClassifier

digits_X, digits_y = load_digits(return_X_y=True)

som = SomClassifier(
    lambda_=76.6,
    n_iter=500,
    sigma_end=0.5,
    random_state=42,
    tau_2=0.01,
)

pipe = Pipeline(steps=[("scaler", StandardScaler()), ("som", som)])
pipe.fit(digits_X, digits_y)

# %%
# Evaluation
# ----------
# Print accuracy and topographic quality metrics.

print(f"Accuracy:           {pipe.score(digits_X, digits_y):.4f}")
print(f"Topographic error:  {som.topographic_error_:.4f}")
print(f"Quantization error: {som.quantization_error_:.4f}")

# %%
# Grid Layout – Neurons by Dominant Class
# ----------------------------------------
# Each neuron colored by its dominant digit class.
# Spatial clustering shows how the SOM organizes the digit space.

som.plot(layout="grid", color="label", palette="Set1").show()

# %%
# PCA Layout
# ----------
# Same coloring but neurons arranged by PCA of their weight vectors.
# Reveals the structure of the learned representation in weight space.

som.plot(layout="pca", color="label", palette="Set1", X=digits_X).show()
