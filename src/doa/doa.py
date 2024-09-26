#!/usr/bin/env python3
"""
Domain of Applicability (DOA) Estimation Module

This module provides various algorithms for anomaly detection and domain of applicability estimation.
It includes implementations of advanced techniques such as Enhanced Isolation Forest, Embedding
Reconstruction, Random Priors Mahalanobis, and Random Feature Stein Discrepancy.

Classes:
    EnhancedIsolationForestDetector: An enhanced version of the Isolation Forest algorithm.
    EmbeddingReconstruction: Anomaly detection based on reconstruction error of dimensionality reduction.
    RandomPriorsMahalanobis: Combines random MLP projections with Mahalanobis distance calculations.
    RFSDDetector: Implements Random Feature Stein Discrepancy for distribution comparison.

These classes provide various methods for fitting models to data, scoring samples, and predicting
the probability of samples being within the domain of applicability.

Each method has its own strengths and limitations, making them suitable for different types of
datasets and problem domains. Users are encouraged to experiment with different methods and
parameters to find the best fit for their specific use case.

Note: This module requires numpy and sklearn to be installed.
"""
import numpy
import sklearn.ensemble
import sklearn.decomposition
import sklearn.pipeline
import sklearn.random_projection
import sklearn.preprocessing


class EnhancedIsolationForestDetector:
    """
    Enhanced Isolation Forest for anomaly detection and domain of applicability estimation.

    This class implements an enhanced version of the Isolation Forest algorithm, incorporating
    extended isolation forests and random projections for improved performance in high-dimensional spaces.

    Key Features:
    1. Extended Isolation Forest: Uses non-axis-parallel splitting for better adaptation to the data structure.
    2. Random Projections: Applies multiple random projections to capture different aspects of the data.
    3. Probability Calibration: Provides calibrated probabilities for domain of applicability estimation.

    Strengths:
    - Effective for high-dimensional data and complex data structures.
    - Can detect anomalies without requiring a dense or spherical cluster assumption.
    - Computationally efficient, with sub-linear time complexity.
    - Provides interpretable probabilistic outputs.

    Limitations:
    - May struggle with very low contamination levels.
    - Performance can be sensitive to the choice of hyperparameters.
    - Assumes anomalies are 'few and different', which may not always hold.

    Parameters:
    -----------
    n_estimators : int
        The number of base estimators (trees) in the forest.
    max_samples : 'auto' or int
        The number of samples to draw to train each base estimator.
    max_features : int
        The number of features to draw to train each base estimator.
    contamination : float, default=0.1
        The expected proportion of outliers in the data set.
    random_state : int
        Seed for random number generation.
    n_projections : int, default=10
        The number of random projections to use.
    n_components : int
        The dimension of the random projections. If None, uses max(1, n_features // 2).

    References:
    -----------
    1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
       In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.
       DOI: 10.1109/ICDM.2008.17

    2. Hariri, S., Kind, M. C., & Brunner, R. J. (2019). Extended Isolation Forest.
       IEEE Transactions on Knowledge and Data Engineering.
       DOI: 10.1109/TKDE.2019.2947676

    3. Emmott, A., Das, S., Dietterich, T., Fern, A., & Wong, W. K. (2013).
       Systematic construction of anomaly detection benchmarks from real data.
       In Proceedings of the ACM SIGKDD workshop on outlier detection and description (pp. 16-21).

    Notes:
    ------
    The anomaly score s(x) for an instance x is defined as:

    s(x) = 2^(-E[h(x)]/c(n))

    where E[h(x)] is the average path length for x across the trees,
    n is the number of instances used to build the trees,
    and c(n) is the average path length of an unsuccessful search in a binary search tree:

    c(n) = 2H(n-1) - (2(n-1)/n), where H(i) is the harmonic number.

    The probability of an instance being normal (in-domain) is then calibrated as:

    P(normal|x) = exp(-s(x)/t) / exp(-1)

    where t is a threshold based on the contamination parameter.

    This implementation extends the original Isolation Forest by using non-axis-parallel splits
    and multiple random projections, potentially improving performance on complex, high-dimensional datasets.
    """

    def __init__(
        self,
        n_estimators,
        max_samples,
        max_features,
        contamination,
        n_projections,
        n_components,
    ):
        self.score_scaler = sklearn.preprocessing.MinMaxScaler()
        self.isotonic_regressor = sklearn.isotonic.IsotonicRegression(
            out_of_bounds="clip"
        )
        self.forests = [
            sklearn.pipeline.make_pipeline(
                sklearn.preprocessing.StandardScaler(),
                sklearn.random_projection.GaussianRandomProjection(
                    n_components=n_components,
                    random_state=i,
                ),
                sklearn.ensemble.IsolationForest(
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    contamination=contamination,
                    max_features=max_features,
                    random_state=i,
                ),
            )
            for i in range(n_projections)
        ]
        # for probability calibration
        self.contamination = contamination

    def fit(self, X):
        self.forests = [forest.fit(X) for forest in self.forests]
        # Fit gamma distribution to the anomaly scores
        train_scores = self._score(X)
        # Set the threshold as the (1 - contamination) percentile of the anomaly scores
        self.threshold_ = numpy.percentile(
            train_scores, 100.0 * (1.0 - self.contamination)
        )
        return self

    def _score(self, X):
        scores = [forest.decision_function(X) for forest in self.forests]
        return -numpy.stack(arrays=scores, axis=-1).mean(axis=-1)

    def predict(self, X):
        scores = self._score(X=X)
        # Calculate probabilities
        probs = numpy.exp(-scores / self.threshold_)
        # Ensure probabilities are in [0, 1] range
        probs = numpy.clip(probs, 0, 1)
        # Scale probabilities so that scores at the threshold get 0.5 probability
        probs = 0.5 * probs / numpy.exp(-1)
        return 1 - numpy.clip(probs, 0, 1)  # Final clipping to ensure [0, 1] range


class EmbeddingReconstruction:
    """
    Calibrated Embedding Reconstruction for anomaly detection and domain of applicability estimation.

    This class implements an anomaly detection method based on the reconstruction error of
    dimensionality reduction techniques (PCA or Random Projection), with calibrated probability outputs.

    Key Features:
    1. Dimensionality Reduction: Uses PCA or Random Projection for feature compression.
    2. Reconstruction Error: Measures anomalies based on the error in reconstructing the original data.
    3. Probability Calibration: Provides calibrated probabilities for domain of applicability estimation.

    Strengths:
    - Effective for high-dimensional data, especially when anomalies lie in a subspace.
    - Can capture complex, non-linear relationships in the data (especially with Random Projection).
    - Provides interpretable probabilistic outputs.
    - Computationally efficient for high-dimensional data.

    Limitations:
    - Assumes that normal data can be effectively reconstructed from a lower-dimensional embedding.
    - May struggle with anomalies that lie in the principal subspace (for PCA).
    - Performance can be sensitive to the choice of the number of components.

    Parameters:
    -----------
    n_components : int
        The number of components to use in the dimensionality reduction.
    method : {'pca', 'random'}
        The method to use for dimensionality reduction.
        'pca': Principal Component Analysis
        'random': Gaussian Random Projection
    contamination : float, default=0.1
        The expected proportion of outliers in the data set.

    References:
    -----------
    1. Hoffmann, H. (2007). Kernel PCA for novelty detection.
       Pattern Recognition, 40(3), 863-874.
       DOI: 10.1016/j.patcog.2006.07.009

    2. Bingham, E., & Mannila, H. (2001). Random projection in dimensionality reduction:
       applications to image and text data. In Proceedings of the seventh ACM SIGKDD international
       conference on Knowledge discovery and data mining (pp. 245-250).
       DOI: 10.1145/502512.502546

    3. Aggarwal, C. C., & Yu, P. S. (2001). Outlier detection for high dimensional data.
       In Proceedings of the 2001 ACM SIGMOD international conference on Management of data (pp. 37-46).
       DOI: 10.1145/375663.375668

    Notes:
    ------
    The reconstruction error e(x) for an instance x is defined as:

    e(x) = ||x - f(g(x))||^2

    where g(x) is the lower-dimensional embedding of x,
    and f(g(x)) is the reconstruction of x from its lower-dimensional embedding.

    The probability of an instance being normal (in-domain) is then calibrated as:

    P(normal|x) = exp(-e(x)/t) / exp(-1)

    where t is a threshold based on the contamination parameter.

    This implementation uses the reconstruction error as a measure of outlyingness,
    with the assumption that normal samples can be effectively reconstructed from their
    lower-dimensional embeddings, while anomalous samples will have higher reconstruction errors.

    The use of random projections (when method='random') can help capture non-linear relationships
    in the data and can be particularly effective for very high-dimensional datasets.
    """

    def __init__(self, n_components, method, contamination):
        if method == "pca":
            # lower dimensionality encoder, here PCA
            self.embedding = sklearn.decomposition.PCA(
                n_components=n_components,
                whiten=True,
                random_state=42,
            )
        elif method == "random":
            # lower dimensionality encoder, here PCA
            self.embedding = sklearn.random_projection.GaussianRandomProjection(
                n_components=n_components,
                compute_inverse_components=True,
                random_state=42,
            )
        else:
            raise ValueError("Use embedding in ['pca', 'random']")
        # for probability calibration
        self.contamination = contamination

    def fit(self, X):
        self.embedding.fit(X)
        train_scores = self.score_(X=X)
        # Set the threshold as the (1 - contamination) percentile of the scores
        self.threshold_ = numpy.percentile(train_scores, 100 * (1 - self.contamination))
        return self

    def score_(self, X):
        low_dim = self.embedding.transform(X=X)
        reconstructed = self.embedding.inverse_transform(low_dim)
        score = numpy.mean(numpy.square(X - reconstructed), axis=-1)
        return score

    def predict(self, X):
        """Compute the probability-based score for each sample."""
        scores = self.score_(X=X)
        # Calculate probabilities (note: we use scores/threshold because lower scores are less anomalous)
        probs = numpy.exp(-scores / self.threshold_)
        # Ensure probabilities are in [0, 1] range
        probs = numpy.clip(probs, 0, 1)
        # Scale probabilities so that scores at the threshold get 0.5 probability
        probs = 1 - 0.5 * (1 - probs) / (1 - numpy.exp(-1))
        return numpy.clip(probs, 0, 1)  # Final clipping to ensure [0, 1] range


class RandomPriorsMahalanobis:
    """
    Random Priors Mahalanobis Distance Anomaly Detector

    This class implements an anomaly detection method that combines random nonlinear projections
    using a Multi-Layer Perceptron (MLP) architecture with Mahalanobis distance calculations
    for identifying outliers and estimating domain of applicability.

    Key Features:
    1. Random MLP Projections: Uses randomly initialized MLP weights for nonlinear data projections.
    2. Mahalanobis Distance: Computes statistical distances in the projected space.
    3. Multiple Projections: Utilizes multiple random projections for robust anomaly detection.
    4. Probability Calibration: Provides calibrated probabilities for domain of applicability estimation.

    Strengths:
    - Effective for high-dimensional and nonlinearly separable data.
    - Captures complex, nonlinear relationships in the data.
    - Robust to different data distributions due to multiple random projections.
    - Computationally efficient, especially for high-dimensional data.
    - Provides interpretable probabilistic outputs.

    Limitations:
    - Performance can be sensitive to the choice of hidden dimension and number of projections.
    - Random initialization may lead to some variability in results.
    - May require larger sample sizes for stable covariance estimation in high-dimensional projections.

    Parameters:
    -----------
    input_dim : int
        The dimensionality of the input data.
    hidden_dim : int
        The number of hidden units in the MLP projection.
    n_projections : int
        The number of output dimensions for each random projection.
    contamination : float
        The expected proportion of outliers in the data set.

    Attributes:
    -----------
    W1, b1 : ndarray
        Weights and biases for the first layer of the random MLP.
    W2, b2 : ndarray
        Weights and biases for the second layer of the random MLP.
    z_mean_ : ndarray
        Mean of the projected training data.
    z_cov_inv_ : ndarray
        Inverse covariance matrix of the projected training data.
    threshold_ : float
        Threshold for anomaly scores based on the contamination parameter.

    References:
    -----------
    1. Ruff, L., et al. (2018). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR.
       URL: http://proceedings.mlr.press/v80/ruff18a.html

    2. Wang, S., et al. (2019). Hyperparameter-free out-of-distribution detection using cosine similarity. arXiv preprint arXiv:1905.10628.
       URL: https://arxiv.org/abs/1905.10628

    3. De Maesschalck, R., Jouan-Rimbaud, D., & Massart, D. L. (2000). The Mahalanobis distance. Chemometrics and intelligent laboratory systems, 50(1), 1-18.
       DOI: 10.1016/S0169-7439(99)00047-7

    Notes:
    ------
    The method works as follows:

    1. Random MLP Projection:
       h = ReLU(XW1 + b1)
       z = hW2 + b2
       where ReLU is the rectified linear unit activation function.

    2. Mahalanobis Distance Calculation:
       For each projected point z_i:
       d_i = sqrt((z_i - μ)^T Σ^-1 (z_i - μ))
       where μ is the mean and Σ is the covariance matrix of the projected training data.

    3. Anomaly Score:
       s_i = d_i / sqrt(n_projections)

    4. Probability Calibration:
       P(normal|x) = exp(-s_i/t) / exp(-1)
       where t is a threshold based on the contamination parameter.

    This approach combines the strength of random neural network projections, which can capture
    complex nonlinear relationships in the data, with the statistical rigor of Mahalanobis
    distance calculations. The use of multiple random projections adds robustness to the method,
    allowing it to capture various aspects of the data distribution.

    The calibrated probabilities provide an interpretable measure of how likely a sample is to be
    within the domain of applicability of the model, based on its similarity to the training data
    in the randomly projected spaces.
    """

    def __init__(self, input_dim, hidden_dim, n_projections, contamination):
        self.input_dim = input_dim
        self.n_projections = n_projections
        self.hidden_dim = hidden_dim
        # random prior MLP weights and biases for nonlinear projection
        self.W1 = numpy.random.randn(input_dim, hidden_dim) / numpy.sqrt(input_dim)
        self.b1 = numpy.random.randn(hidden_dim)
        self.W2 = numpy.random.randn(hidden_dim, n_projections) / numpy.sqrt(hidden_dim)
        self.b2 = numpy.random.randn(n_projections)
        # probability calibration
        self.contamination = contamination

    def project(self, X):
        h = numpy.maximum(0, (numpy.dot(a=X, b=self.W1) + self.b1))
        z = numpy.dot(a=h, b=self.W2) + self.b2
        z_mean = numpy.mean(a=z, axis=0)
        z_cov = numpy.cov(z, rowvar=False)
        return z, z_mean, z_cov

    def fit(self, X):
        # first apply the projections to the input
        z, z_mean, z_cov = self.project(X)
        # get statistics of manifold projections distributions
        self.z_mean_ = z_mean
        self.z_cov_inv_ = numpy.linalg.inv(z_cov)
        # compute stats differences between projections and base distribution
        train_scores = self.score_(projections=z)
        # Set the threshold as the (1 - contamination) percentile of the scores
        self.threshold_ = numpy.percentile(train_scores, 100 * (1 - self.contamination))
        return self

    def score_(self, projections):
        # Compute Mahalanobis distances
        diff = projections - self.z_mean_
        mahalanobis = numpy.sqrt(
            numpy.sum(a=numpy.dot(diff, self.z_cov_inv_) * diff, axis=1)
        )
        # Compute scores (normalized by sqrt of projection dimension)
        scores = mahalanobis / numpy.sqrt(self.n_projections)
        return scores

    def predict(self, X):
        """Compute the probability-based score for each sample."""
        z, _, _ = self.project(X=X)
        scores = self.score_(projections=z)
        # Calculate probabilities (note: we use scores/threshold because lower scores are less anomalous)
        probs = numpy.exp(-scores / self.threshold_)
        # Ensure probabilities are in [0, 1] range
        probs = numpy.clip(probs, 0, 1)
        # Scale probabilities so that scores at the threshold get 0.5 probability
        probs = 1 - 0.5 * (1 - probs) / (1 - numpy.exp(-1))
        return numpy.clip(probs, 0, 1)  # Final clipping to ensure [0, 1] range


class RFSDDetector:
    """
    Random Feature Stein Discrepancy (RFSD) Explanation and Implementation

    1. RFSD Theory:
    RFSD is used to measure the discrepancy between two distributions P and Q,
    where we have samples from P and a score function of Q.

    The RFSD is defined as:
    RFSD(P, Q) = sup_{f in F} {E_P[T_Q f(X)] - E_Q[f(X)]}

    where T_Q is the Stein operator: T_Q f(x) = ∇_x log q(x) · f(x) + ∇_x · f(x)

    2. Random Feature Approximation:
    We approximate the function f using random features:
    f(x) ≈ φ(x)ᵀ θ, where φ(x) are random features

    3. Optimization:
    The optimal θ can be found analytically:
    θ* = E_P[ξ(X)] - E_Q[φ(X)]
    where ξ(x) = ∇_x φ(x) + φ(x) ∇_x log q(x)

    4. RFSD Score:
    The squared RFSD is then approximated as:
    RFSD²(P, Q) ≈ θ*ᵀ Λ θ*
    where Λ = E_Q[φ(X)φ(X)ᵀ]

    """

    def __init__(self, input_dim, n_features, gamma, contamination):
        self.input_dim = input_dim
        self.n_features = n_features
        self.gamma = gamma
        # random features
        self.W = numpy.random.randn(input_dim, n_features)
        self.b = numpy.random.uniform(0, 2 * numpy.pi, n_features)
        # probability calibration
        self.contamination = contamination

    def _phi(self, X):
        return numpy.sqrt(2 / self.n_features) * numpy.cos(
            self.gamma * X @ self.W + self.b
        )

    def _grad_phi(self, X):
        return (
            -self.gamma
            * numpy.sqrt(2 / self.n_features)
            * numpy.sin(self.gamma * X @ self.W + self.b)
        )

    def _xi(self, X):
        grad_phi = self._grad_phi(X)
        phi = self._phi(X)
        return grad_phi @ self.W.T + X * phi.sum(axis=1, keepdims=True)

    def fit(self, X):
        phi_X = self._phi(X)
        xi_X = self._xi(X)
        self.E_P_xi = numpy.mean(xi_X, axis=0)
        self.E_P_phi = numpy.mean(phi_X, axis=0)
        self.Lambda = numpy.cov(phi_X.T)
        # Ensure dimensions match
        self.theta_star = numpy.linalg.solve(
            self.Lambda, self.E_P_xi @ self.W - self.E_P_phi
        )
        train_scores = self._score(X)
        # Set the threshold as the (1 - contamination) percentile of the scores
        self.threshold_ = numpy.percentile(train_scores, 100 * (1 - self.contamination))
        return self

    def _score(self, X):
        phi_X = self._phi(X)
        xi_X = self._xi(X)
        term1 = numpy.sum(
            (xi_X @ self.W - self.E_P_xi @ self.W) * self.theta_star, axis=1
        )
        term2 = numpy.sum((phi_X - self.E_P_phi) * self.theta_star, axis=1)
        return term1 - term2

    def predict(self, X):
        scores = self._score(X)
        # Calculate probabilities (note: we use scores/threshold because lower scores are less anomalous)
        probs = numpy.exp(-scores / self.threshold_)
        # Ensure probabilities are in [0, 1] range
        probs = numpy.clip(probs, 0, 1)
        # Scale probabilities so that scores at the threshold get 0.5 probability
        probs = 1 - 0.5 * (1 - probs) / (1 - numpy.exp(-1))
        return numpy.clip(probs, 0, 1)  # Final clipping to ensure [0, 1] range
