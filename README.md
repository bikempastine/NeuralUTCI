# utci-nn

A neural network emulator for the Universal Thermal Climate Index (UTCI), as described in Pastine et al.

Model is trained on 'Grid data' from Brode et al., 2012 and can be downloaded:
https://link.springer.com/article/10.1007/s00484-011-0454-1

## Installation

pip install git+https://github.com/bikempastine/utci-nn.git


## Usage
```python
from utci_nn import NN_UTCI

result = NN_UTCI(Ta=20, Tr=25, va=1.0, rH=50)
```

### Parameters

| Parameter | Description | Units | Valid Range |
|---|---|---|---|
| `Ta` | Air temperature | °C | -50 to 50 |
| `Tr` | Mean radiant temperature | °C | -80 to 120 |
| `va` | Wind speed at 10 m | m/s | 0.5 to 30.3 |
| `rH` | Relative humidity | % | 5 to 100 |

### Out-of-bounds handling
```python
# Returns NaN for out-of-bounds values (default)
result = NN_UTCI(Ta=20, Tr=25, va=1.0, rH=50, oob="nan")

# Clamps inputs to valid range
result = NN_UTCI(Ta=20, Tr=25, va=1.0, rH=50, oob="clamp")
```

## Reference

Pastine et al. ...
Bröde et al. (2012). Int. J. Biometeorology, 56(3), 481-494.
