# NeuralUTCI

A neural network emulator for the **Universal Thermal Climate Index (UTCI)** — a biometeorological index that quantifies human thermal comfort from standard meteorological inputs.

UTCI integrates air temperature, mean radiant temperature, wind speed, and relative humidity into a single equivalent temperature (°C) that reflects how the human body actually perceives the thermal environment. It is widely used in public health, climate services, and urban planning.

This package provides a pre-trained **4-layer MLP** (4 → 79 → 75 → 39 → 1) as a drop-in replacement for the standard 6th-degree polynomial approximation of Bröde et al. (2012). The neural network is trained on the same input domain and is designed to be faster and more accurate, especially near the boundaries of the valid input range.

> **Note:** This package is associated with a conference contribution (EGU 2026) and has not yet undergone formal peer review.

## Installation

```bash
pip install git+https://github.com/bikempastine/NeuralUTCI.git
```

**Dependencies:** `torch`, `numpy` — no pandas or scikit-learn required at runtime.

> PyTorch is a large dependency (~500 MB for CPU-only). If you only need inference on small arrays, the install is still straightforward but be aware of the download size.

## Quick start

```python
from NeuralUTCI import utci

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

pandas `Series` inputs are also accepted and are automatically converted to numpy arrays.

## Parameters

| Parameter | Description | Units | Valid range |
|---|---|---|---|
| `Ta` | Air temperature | °C | −50 to +50 |
| `Tr` | Mean radiant temperature | °C | −80 to +120 |
| `va` | Wind speed at 10 m height | m/s | 0.5 to 30.3 |
| `rH` | Relative humidity | % | 5 to 100 |
| `oob` | Out-of-bounds strategy | — | `"nan"` (default) or `"clamp"` |

All four meteorological inputs must have the same shape. Scalar and array inputs can be mixed freely.

## Out-of-bounds handling

The model is trained on the same input domain as the polynomial approximation in Bröde et al. (2012). Inputs outside this domain are handled with the `oob` parameter:

- `oob="nan"` *(default)* — any row with one or more out-of-bounds inputs returns `NaN`
- `oob="clamp"` — inputs are clamped to the valid range before prediction, matching the behaviour described in Bröde et al. (2012)

```python
# Returns NaN where any input is out of bounds
result = utci(Ta=np.array([25.0, 99.0]), Tr=np.array([30.0, 99.0]),
              va=np.array([2.5, 2.5]), rH=np.array([50.0, 50.0]), oob="nan")
# → [28.4, nan]

# Clamps out-of-bounds inputs to the valid range before predicting
result = utci(Ta=np.array([25.0, 99.0]), Tr=np.array([30.0, 99.0]),
              va=np.array([2.5, 2.5]), rH=np.array([50.0, 50.0]), oob="clamp")
# → [28.4, <value at clamped inputs>]
```

## Model architecture

The emulator is a fully connected feedforward neural network:

```
Input (4) → Linear(4→79) → ReLU → Linear(79→75) → ReLU → Linear(75→39) → ReLU → Linear(39→1)
```

Inputs are pre-scaled using a `ColumnTransformer`:
- `StandardScaler` on `Ta` and `Tr − Ta`
- `MinMaxScaler` on `va` and `rH`

The network predicts the **UTCI offset** (UTCI − Ta), and the final UTCI value is recovered by adding `Ta`. This decomposition reduces the dynamic range the network must learn.

Weights are stored in `assets/utci_nn_weights.pth` (PyTorch state dict, ~40 KB). Scaler parameters are stored in `assets/scaler_params.json` for transparency and version stability.

## Training data

The model is trained on the 'Grid data' from Bröde et al. (2012), which can be downloaded from the paper supplementary materials:
https://link.springer.com/article/10.1007/s00484-011-0454-1

## References

Pastine, B., Klöwer, M., Tang, T., Wilson Kemsley, S., and Slater, L.: An upgraded neural network-based operational procedure for the Universal Thermal Climate Index (UTCI), EGU General Assembly 2026, Vienna, Austria, 3–8 May 2026, EGU26-5838, https://doi.org/10.5194/egusphere-egu26-5838, 2026.

Bröde, P., Fiala, D., Błażejczyk, K. et al. Deriving the operational procedure for the Universal Thermal Climate Index (UTCI). Int. J. Biometeorol. 56, 481–494 (2012). https://doi.org/10.1007/s00484-011-0454-1

## License

CC0-1.0 — see [LICENSE](LICENSE).
