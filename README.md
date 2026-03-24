# NeuralUTCI

A neural network emulator for the Universal Thermal Climate Index (UTCI), as described in Pastine et al. (2026)

Model is trained on 'Grid data' from Bröde et al (2012) and can be downloaded:
https://link.springer.com/article/10.1007/s00484-011-0454-1

> **Note:** This package is associated with a conference contribution (EGU 2026) and has not yet undergone peer review.

## Installation
```bash
pip install git+https://github.com/bikempastine/NeuralUTCI.git
```

## Usage
```python
from NeuralUTCI import utci

result = utci(Ta=20, Tr=25, va=1.0, rH=50)
# Scalar inputs — returns a float
result = utci(Ta=25, Tr=30, va=2.5, rH=50)
print(result)  # e.g. 28.4 °C

# Array inputs — returns a numpy array
import numpy as np
Ta = np.array([10.0, 25.0, 35.0])
Tr = np.array([15.0, 30.0, 45.0])
va = np.array([1.0,  2.5,  0.5])
rH = np.array([60.0, 50.0, 80.0])
result = utci(Ta=Ta, Tr=Tr, va=va, rH=rH)
```

**Dependencies:** `torch`, `numpy`

### Parameters

| Parameter | Description | Units | Valid Range |
|---|---|---|---|
| `Ta` | Air temperature | °C | −50 to +50 |
| `Tr` | Mean radiant temperature | °C | −80 to +120 |
| `va` | Wind speed at 10 m height | m/s | 0.5 to 30.3 |
| `rH` | Relative humidity | % | 5 to 100 |
| `oob` | Out-of-bounds strategy | — | `"nan"` (default) or `"clamp"` |

### Out-of-bounds handling

The NN is trained on the same input domain as the polynomial approximation in Bröde et al. (2012).
Inputs outside this domain are handled as indicated by the `oob` parameter:

- `oob="nan"` *(default)* — any row with one or more out-of-bounds inputs returns `NaN`
- `oob="clamp"` — inputs are clamped to the valid range before prediction, matching the behaviour described in Bröde et al. (2012)

```python
# Returns NaN for out-of-bounds inputs (default)
result = utci(Ta=20, Tr=25, va=1.0, rH=50, oob="nan")
# Returns NaN where any input is out of bounds
result = utci(Ta=np.array([25.0, 99.0]), Tr=np.array([30.0, 99.0]),
              va=np.array([2.5, 2.5]), rH=np.array([50.0, 50.0]), oob="nan")
# → [28.4, nan]

# Clamps out-of-bounds inputs to the valid range before predicting
result = utci(Ta=np.array([25.0, 99.0]), Tr=np.array([30.0, 99.0]),
              va=np.array([2.5, 2.5]), rH=np.array([50.0, 50.0]), oob="clamp")
# → [28.4, <value at clamped inputs>]
```

## Requirements
Nerual network weights are stored in `assets/utci_nn_weights.pth` (PyTorch state dict, ~40 KB). Scaler parameters are stored in `assets/scaler_params.json` for transparency and version stability.

## Training data

The model is trained on the 'Grid data' from Bröde et al. (2012), which can be downloaded from the paper supplementary materials:
https://link.springer.com/article/10.1007/s00484-011-0454-1

## References

Pastine, B., Klöwer, M., Tang, T., Wilson Kemsley, S., and Slater, L.: An upgraded neural network-based operational procedure for the Universal Thermal Climate Index (UTCI), EGU General Assembly 2026, Vienna, Austria, 3–8 May 2026, EGU26-5838, https://doi.org/10.5194/egusphere-egu26-5838, 2026.

Bröde, P., Fiala, D., Błażejczyk, K. et al. Deriving the operational procedure for the Universal Thermal Climate Index (UTCI). Int. J. Biometeorol. 56, 481–494 (2012). https://doi.org/10.1007/s00484-011-0454-1


## Cite this Package

Pastine, B., Klöwer, M., Tang, T., Wilson Kemsley, S., & Slater, L. (2026). NeuralUTCI (Version 0.0.2) [Computer software]. https://doi.org/10.5281/zenodo.19204911

## License
CC0-1.0 — see [LICENSE](LICENSE).
