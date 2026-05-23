User Guide
=====================
Both the SomClassifier and SomVQ implement the scikit-learn API and can be used as drop in replacements for other scikit-learn Estimators. 

Classification
--------------

.. code-block:: python
    from dbgsom.SomClassifier import SomClassifier
    from sklearn.datasets import load_digits
    digits_X, digits_y = load_digits(return_X_y=True)

    classifier.fit(digits_X, digits_y)
    classifier.score(digits_X, digits_y)

.. code-block::
    0.8375069560378409

.. code-block:: python
    classifier.predict(digits_X)

.. code-block::
    array([0, 1, 8, ..., 8, 9, 6], shape=(1797,))

Clustering
----------