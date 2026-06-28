from __future__ import annotations

"""Deformable-registration d2 toolkit (isolated from d2_methods.py to survive
parallel edits). Holds the functions behind the best submission (0.91981):
deformable (Demons) template normalization + per-slab features + fusion helpers.

Depends only on stable helpers in d2_methods: _register_to_template, flat_feature,
_fit_pca_ridge, _l2.
"""

import numpy as np
from scipy.ndimage import zoom

from d2_methods import _register_to_template, flat_feature, _fit_pca_ridge, _l2


# --- deformable (elastic) registration ---------------------------------------

def _deformable_to_template(vol, tpl, smooth=3.0, iters=40):
    """Regularized symmetric Demons deformable registration to template (same
    modality). High `smooth` => gentler field => subject anatomy preserved."""
    import SimpleITK as sitk
    fixed = sitk.GetImageFromArray(np.ascontiguousarray(tpl, dtype=np.float32))
    moving = sitk.GetImageFromArray(np.ascontiguousarray(vol, dtype=np.float32))
    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(iters)
    demons.SetStandardDeviations(smooth)
    try:
        disp = demons.Execute(fixed, moving)
    except Exception:
        return vol
    tx = sitk.DisplacementFieldTransform(disp)
    out = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(out)


def _deform_norm(vol, tpl, smooth=3.0):
    return _deformable_to_template(_register_to_template(vol, tpl), tpl, smooth=smooth)


def _bspline_to_template(vol, tpl, nodes=6, iters=40):
    """Multi-resolution B-spline FFD deformable (pyramid 4->2->1). `nodes` =
    control-grid size; fewer = smoother/more regularized."""
    import SimpleITK as sitk
    fixed = sitk.GetImageFromArray(np.ascontiguousarray(tpl, dtype=np.float32))
    moving = sitk.GetImageFromArray(np.ascontiguousarray(vol, dtype=np.float32))
    tx = sitk.BSplineTransformInitializer(fixed, [nodes] * 3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=iters)
    R.SetInitialTransform(tx, inPlace=True)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    try:
        R.Execute(fixed, moving)
    except Exception:
        return vol
    out = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(out)


def _bspline_norm(vol, tpl, nodes=6):
    return _bspline_to_template(_register_to_template(vol, tpl), tpl, nodes=nodes)


# --- per-slab (per-height) features ------------------------------------------

def _slab_feats(vol, K=10, s=16):
    G = vol.shape[0]
    edges = np.linspace(0, G, K + 1).astype(int)
    out = []
    for k in range(K):
        a, b = edges[k], max(edges[k] + 1, edges[k + 1])
        slab = vol[a:b].mean(axis=0)
        z = zoom(slab, s / G, order=1) if slab.shape[0] != s else slab
        out.append(_l2(z.reshape(-1).astype(np.float32)))
    return np.stack(out)


# --- score-matrix helpers ----------------------------------------------------

def _zscore_rows(s):
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-6)


def _sinkhorn(S, tau=0.05, iters=50):
    P = np.exp((S - S.max()) / tau).astype(np.float64)
    for _ in range(iters):
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        P = P / (P.sum(axis=0, keepdims=True) + 1e-12)
    return P.astype(np.float32)
