import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from typing import Union, Literal

# ── Neural network model definition ───────────────────────────────────────────
class UTCI_NN_Emulator(nn.Module):
    """
    Neural network emulator for Universal Thermal Climate Index (UTCI).
    """
    def __init__(self):
        super(UTCI_NN_Emulator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 79),
            nn.ReLU(),
            nn.Linear(79, 75),
            nn.ReLU(),
            nn.Linear(75, 39),
            nn.ReLU(),
            nn.Linear(39, 1)
        )

    def forward(self, x):
        return self.model(x)


# ── Load assets at import time ───────────────────────────────────────────────────
_ASSETS_DIR = Path(__file__).parent / "assets"
_model = UTCI_NN_Emulator()
_model.load_state_dict(torch.load(_ASSETS_DIR / "utci_nn_weights.pth", weights_only=True))
_model.eval()

with open(_ASSETS_DIR / "scaler_params.json") as _f:
    _sp = json.load(_f)
_STD_MEAN  = np.array(_sp["standard_mean"],  dtype=np.float64)
_STD_SCALE = np.array(_sp["standard_scale"], dtype=np.float64)
_MM_MIN    = np.array(_sp["minmax_min"],     dtype=np.float64)
_MM_MAX    = np.array(_sp["minmax_max"],     dtype=np.float64)


def _scale(Ta, Tr_Ta, va, rH):
    """Apply the ColumnTransformer scaling: StandardScaler on (Ta, Tr-Ta), MinMaxScaler on (va, rH)."""
    Ta_s    = (Ta    - _STD_MEAN[0]) / _STD_SCALE[0]
    TrTa_s  = (Tr_Ta - _STD_MEAN[1]) / _STD_SCALE[1]
    va_s    = (va    - _MM_MIN[0])   / (_MM_MAX[0] - _MM_MIN[0])
    rh_s    = (rH    - _MM_MIN[1])   / (_MM_MAX[1] - _MM_MIN[1])
    return np.column_stack([Ta_s, TrTa_s, va_s, rh_s])


# ── Valid input bounds, from training data ────────────────────────────────────
_BOUNDS = {
    "Ta":    (-50.0,  50.0),
    "Tr":   (-80.0, 120.0),
    "va":    (  0.5,  30.3),
    "rH":    (  5.0, 100.0),
}

# ── Main prediction function ─────────────────────────────────────────────────
def utci(
    Ta: Union[float, np.ndarray],
    Tr: Union[float, np.ndarray],
    va: Union[float, np.ndarray],
    rH: Union[float, np.ndarray],
    oob: Literal["nan", "clamp"] = "nan",
) -> np.ndarray:
    """
    Calculate UTCI using a pre-trained neural network approximator as described in Pastine et al.
    NN is trained on the same data as the polynomial approximation in Bröde et al. (2012) and covers the same input domain.

    Accepts scalars, numpy arrays, or pandas Series for all meteorological inputs.

    Parameters
    ----------
    Ta  : Air temperature (°C),          valid range: -50 to +50
    Tr  : Mean radiant temperature (°C), valid range: -80 to +120
    va  : Wind speed at 10 m (m/s),      valid range: 0.5 to 30.3
    rH  : Relative humidity (%),         valid range: 5 to 100
    oob : Out-of-bounds handling strategy:
            "nan"   – return NaN for any out-of-bounds row (default)
            "clamp" – clamp each variable to its valid range before predicting

    Returns:
    -------
    np.ndarray
        UTCI values in °C defined as UTCI offset plus Air temperature. Shape matches the input arrays.

    References
    ----------
    Bröde, P., Fiala, D., Bla˙zejczyk, K. et al. Deriving the operational procedure
    for the Universal Thermal Climate Index (UTCI). Int. J. Biometeorol. 56, 481–494
    (2012).
    """
    if oob not in ("nan", "clamp"):
        raise ValueError(f"oob must be 'nan' or 'clamp', got '{oob}'")

    # ── Coerce all inputs to float32 numpy arrays ──────────────────────────────
    Ta    = np.asarray(Ta,    dtype=np.float32)
    Tr    = np.asarray(Tr,    dtype=np.float32)
    va    = np.asarray(va,    dtype=np.float32)
    rH    = np.asarray(rH,    dtype=np.float32)

    # ── Keep track if inputs were scalars to return scalar output ──────────────
    scalar_input = Ta.ndim == 0
    Ta, Tr, va, rH = (np.atleast_1d(a) for a in (Ta, Tr, va, rH))

    # ── Check that all inputs have the same length and issue warning ────────────
    shapes = {len(np.atleast_1d(a)) for a in (Ta, Tr, va, rH)}
    if len(shapes) > 1:
        raise ValueError(f"All inputs must have the same length, got shapes: {shapes}")


    # ── Run inference ───────────────────────────────────────────────────────────
    utci = np.full(len(Ta), np.nan)

    if oob == "nan":
        oob_mask = (
            (Ta < _BOUNDS["Ta"][0]) | (Ta > _BOUNDS["Ta"][1])
            | (Tr < _BOUNDS["Tr"][0]) | (Tr > _BOUNDS["Tr"][1])
            | (va < _BOUNDS["va"][0]) | (va > _BOUNDS["va"][1])
            | (rH < _BOUNDS["rH"][0]) | (rH > _BOUNDS["rH"][1])
        )
        valid_mask = ~oob_mask

        if not valid_mask.any():
            return utci[0] if scalar_input else utci  # early return, all NaN

        Tr_Ta = Tr - Ta
        X_scaled = _scale(Ta[valid_mask], Tr_Ta[valid_mask], va[valid_mask], rH[valid_mask])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            predictions = _model(X_tensor).squeeze(-1).numpy()
        utci[valid_mask] = predictions + Ta[valid_mask]  # only write valid rows

    elif oob == "clamp":
        Ta = np.clip(Ta, *_BOUNDS["Ta"])
        Tr = np.clip(Tr, *_BOUNDS["Tr"])
        va = np.clip(va, *_BOUNDS["va"])
        rH = np.clip(rH, *_BOUNDS["rH"])

        Tr_Ta = Tr - Ta
        X_scaled = _scale(Ta, Tr_Ta, va, rH)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            predictions = _model(X_tensor).squeeze(-1).numpy()
        utci[:] = predictions + Ta  # all rows are valid after clamping

    return utci[0] if scalar_input else utci
