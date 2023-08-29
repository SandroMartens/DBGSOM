# Generated by CodiumAI
from dbgsom.dbgsom_ import DBGSOM
import numpy

from dbgsom.dbgsom_ import exponential_decay


import pytest


class TestExponentialDecay:
    # Test that the exponential_decay function returns the expected sigma value when given valid inputs for sigma_start, sigma_end, max_iter, current_iter, and learning_rate.
    def test_behaviour(self):
        sigma_start = 0.2
        sigma_end = 0.05
        max_iter = 100
        current_iter = 50
        learning_rate = 0.01

        expected_sigma = 0.125

        assert (
            exponential_decay(
                sigma_start, sigma_end, max_iter, current_iter, learning_rate
            )
            == expected_sigma
        )


class TestDBGSOM:
    # Test that the DBGSOM model can handle different distance metrics and produce accurate results.
    def test_distance_metrics(self):
        # Create a DBGSOM instance
        som = DBGSOM()

        # Generate some sample data
        X = np.random.rand(100, 10)

        # Test with different distance metrics
        metrics = ["euclidean", "manhattan", "cosine"]
        for metric in metrics:
            som.metric = metric
            som.fit(X)
            labels = som.predict(X)

            # Assert that the labels are unique
            assert len(np.unique(labels)) == len(som.classes_)
