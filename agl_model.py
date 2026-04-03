import os

import numpy as np

from examples.seismic.model import SeismicModel

__all__ = ['agl_model']


def Gardners(vp, normalize=True):
    """
    Gardner's relation for vp in km/s
    """
    b = 1 / (0.31 * (1e3*vp)**0.25)
    if normalize:
        b[vp < 1.51] = 1.0
    return b


def agl_model(preset, **kwargs):
    space_order = kwargs.pop('space_order', 2)
    shape = kwargs.pop('shape', (101, 101))
    spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
    origin = kwargs.pop('origin', tuple([0. for _ in shape]))
    nbl = kwargs.pop('nbl', 10)
    dtype = kwargs.pop('dtype', np.float32)
    vp = kwargs.pop('vp', 1.5)
    nlayers = kwargs.pop('nlayers', 3)
    fs = kwargs.pop('fs', False)
    density = kwargs.pop('density', False)
    if preset.lower() in ['constant-viscoelastic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 2.2 km/s.
        qp = kwargs.pop('qp', 100.)
        vs = kwargs.pop('vs', 1.2)
        qs = kwargs.pop('qs', 70.)
        b = 1/2.

        return SeismicModel(space_order=space_order, vp=vp, qp=qp, vs=vs,
                            qs=qs, b=b, origin=origin, shape=shape,
                            dtype=dtype, spacing=spacing, nbl=nbl,
                            **kwargs)
    elif preset.lower() in ['marmousi-agl-vp', 'marmousi2d-agl-vp']:
        shape = (3401, 701)
        spacing = (5.0, 5.0)
        origin = (0., 0.)
        nbl = kwargs.pop('nbl', 20)
        bcs = kwargs.pop('bcs', 'mask')

        data_path = kwargs.get('data_path', None)
        if data_path is None:
            raise ValueError("Path to devitocodes/data not found! Please specify with "
                         "'data_path=<path/to/devitocodes/data>'")

        vp_path = os.path.join(data_path, 'marmousi-ii_nx3401_nz701_dxdz5m_vp.bin')
        rho_path = os.path.join(data_path, 'marmousi-ii_nx3401_nz701_dxdz5m_density.bin')

        vp = np.fromfile(vp_path, dtype='float32').reshape(shape) / 1000.0  # convert to km/s
        rho = np.fromfile(rho_path, dtype='float32').reshape(shape)

        b = (1.0 / rho).astype(np.float32)

        qp = np.empty(shape, dtype=np.float32)
        qp[:] = 3.516 * ((vp[:] * 1000.0) ** 2.2) * 1e-6   # Li empirical formula
        qp = qp.astype(np.float32)

        return SeismicModel(space_order=space_order,
                        vp=vp,
                        b=b,
                        qp=qp,
                        origin=origin,
                        shape=shape,
                        dtype=np.float32,
                        spacing=spacing,
                        nbl=nbl,
                        bcs='mask',
                        **kwargs)
    else:
        raise ValueError("Unknown model preset name")