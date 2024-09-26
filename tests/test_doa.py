import numpy as np
import pytest
from doa import (
    EnhancedIsolationForestDetector,
    EmbeddingReconstruction,
    RandomPriorsMahalanobis,
    RFSDDetector,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    return X


def test_enhanced_isolation_forest_detector(sample_data):
    detector = EnhancedIsolationForestDetector(
        n_estimators=100,
        max_samples="auto",
        max_features=1,
        contamination=0.1,
        n_projections=10,
        n_components=3,
    )
    detector.fit(sample_data)
    predictions = detector.predict(sample_data)

    assert predictions.shape == (100,)
    assert np.all((predictions >= 0) & (predictions <= 1))


def test_embedding_reconstruction(sample_data):
    for method in ["pca", "random"]:
        detector = EmbeddingReconstruction(
            n_components=3, method=method, contamination=0.1
        )
        detector.fit(sample_data)
        predictions = detector.predict(sample_data)

        assert predictions.shape == (100,)
        assert np.all((predictions >= 0) & (predictions <= 1))


def test_random_priors_mahalanobis(sample_data):
    detector = RandomPriorsMahalanobis(
        input_dim=5, hidden_dim=10, n_projections=3, contamination=0.1
    )
    detector.fit(sample_data)
    predictions = detector.predict(sample_data)

    assert predictions.shape == (100,)
    assert np.all((predictions >= 0) & (predictions <= 1))


def test_rfsd_detector(sample_data):
    detector = RFSDDetector(input_dim=5, n_features=10, gamma=0.1, contamination=0.1)
    detector.fit(sample_data)
    predictions = detector.predict(sample_data)

    assert predictions.shape == (100,)
    assert np.all((predictions >= 0) & (predictions <= 1))


def test_invalid_input():
    with pytest.raises(ValueError):
        EmbeddingReconstruction(
            n_components=3, method="invalid_method", contamination=0.1
        )


def test_contamination_range():
    X = np.random.randn(100, 5)
    with pytest.raises(ValueError):
        EnhancedIsolationForestDetector(
            n_estimators=100,
            max_samples="auto",
            max_features=1,
            contamination=1.5,  # Invalid contamination value
            n_projections=10,
            n_components=3,
        ).fit(X)


if __name__ == "__main__":
    pytest.main()
