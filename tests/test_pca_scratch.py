"""
Unit tests for PCAFromScratch.
"""

import numpy as np
import pytest
from src.pca.pca_scratch import PCAFromScratch


def make_data(n_samples=100, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def test_fit_transform_shape():
    X = make_data()
    pca = PCAFromScratch(n_components=2)
    X_t = pca.fit_transform(X)
    assert X_t.shape == (100, 2)


def test_explained_variance_ratio_leq_one():
    X = make_data()
    pca = PCAFromScratch(n_components=3)
    pca.fit(X)
    assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9


def test_components_shape():
    X = make_data(n_features=5)
    pca = PCAFromScratch(n_components=3)
    pca.fit(X)
    assert pca.components_.shape == (3, 5)


def test_mean_subtraction():
    X = make_data()
    pca = PCAFromScratch(n_components=2)
    pca.fit(X)
    np.testing.assert_allclose(pca.mean_, X.mean(axis=0))
