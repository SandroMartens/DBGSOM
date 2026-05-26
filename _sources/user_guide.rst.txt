User Guide
=====================
Both the SomClassifier and SomVQ implement the scikit-learn API and can be used as drop in replacements for other scikit-learn Estimators. 

Classification
--------------

.. code-block:: python

    from dbgsom.SomClassifier import SomClassifier
    from sklearn.datasets import load_digits
    digits_X, digits_y = load_digits(return_X_y=True)

    classifier = SomClassifier()
    classifier.fit(digits_X, digits_y)
    classifier.score(digits_X, digits_y)

.. code-block:: pycon

    >>> 0.8375069560378409

.. code-block:: python

    classifier.predict(digits_X)

.. code-block:: pycon

    >>> array([0, 1, 8, ..., 8, 9, 6], shape=(1797,))

Clustering
----------
.. code-block:: python

    from dbgsom.SomVQ import SomVQ
    from sklearn.datasets import load_digits
    digits_X, digits_y = load_digits(return_X_y=True)

    som = SomVQ()
    som.fit(digits_X)

.. code-block:: python

    som.predict(digits_X)

.. code-block:: pycon

    >>> array([6, 0, 1, ..., 0, 4, 3], shape=(1797,))
    
.. code-block:: python

    som.quantization_error_

.. code-block:: pycon

    >>> 24.360118119212867
