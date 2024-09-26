# Domain of Applicability

DOA is a Python package for anomaly detection and domain of applicability estimation. It includes implementations of advanced techniques such as Enhanced Isolation Forest, Embedding Reconstruction, Random Priors Mahalanobis, and Random Feature Stein Discrepancy.

## Installation

You will soon be able to install the DOA Estimator package using pip:

```
pip install doa
```

## Usage

Here's a quick example of how to use the EnhancedIsolationForestDetector:

```python
from doa import EnhancedIsolationForestDetector
import numpy as np

# Generate some random data
X = np.random.randn(100, 5)

# Create and fit the detector
detector = EnhancedIsolationForestDetector(
    n_estimators=100,
    max_samples='auto',
    max_features=1,
    contamination=0.1,
    n_projections=10,
    n_components=3
)
detector.fit(X)

# Make predictions
predictions = detector.predict(X)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
