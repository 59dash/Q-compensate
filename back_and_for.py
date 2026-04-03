"""
Viscoacoustic RTM with symbolic forward + backward operators
=============================================================
Both forward and backward propagation are built from the viscoacoustic
equations in operators.py, matching SLS/KV/Maxwell 2nd order.

Snapshots are saved to disk with ZFP compression during forward,
then loaded during backward for the imaging condition.
"""

import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# DevitoPRO path (comment out if using standard devito)
devitopro_dir = "/home/xlz/Python/devitopro/devitopro-trial-ou"
devito_dir = devitopro_dir + "/submodules/devito"
sys.path.insert(0, devito_dir)
sys.path.insert(0, devitopro_dir)

from devito import (TimeFunction, Function, Operator, Eq, solve,
                    norm, gaussian_smooth, div, grad, NODE)
from devito import configuration
from examples.seismic import (AcquisitionGeometry, PointSource,
                               plot_image, plot_velocity, demo_model)
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver

import zfpy

configuration['log-level'] = 'DEBUG'

from agl_model import agl_model



# Model

flag_model = 'marmousi'

match flag_model:
    case 'demo':
        def create_model(grid=None):
            model = demo_model('layers-viscoacoustic', origin=(0., 0.),
                                shape=(101, 101), spacing=(10., 10.),
                                nbl=20, grid=grid, nlayers=2)
            return model
    case 'marmousi':
        def create_model(grid=None):
            model = agl_model('marmousi-agl-vp',
                              data_path="/home/xlz/marmousi",
                              grid=grid, nbl=100)
            return model

filter_sigma = (1, 1)
nshots = 20
nreceivers = 1701
t0 = 0.
tn = 4000.
f0 = 0.010
SO = 4  # space order
timesteps_IC = 4  # snapshot save interval

model = create_model()
model0 = create_model(grid=model.grid)
gaussian_smooth(model0.vp, sigma=filter_sigma)

# ==============================================================
# Geometry
# ==============================================================
src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, -1] = 20.

rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nreceivers)
rec_coordinates[:, 1] = 30.

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                t0, tn, f0=f0, src_type='Ricker')

source_locations = np.empty((nshots, 2), dtype=np.float32)
source_locations[:, 0] = np.linspace(0., model.domain_size[0], num=nshots)
source_locations[0, 1] = 30.

mute_samples = int(200 / model0.critical_dt)


#saveload snapshot with ZFP compression
def save_snapshot(data, filepath, precision=16):
    with open(filepath, 'wb') as f:
        f.write(zfpy.compress_numpy(data, precision=precision))

def load_snapshot(filepath):
    with open(filepath, 'rb') as f:
        return zfpy.decompress_numpy(f.read())

def snap_filename(snapDir, model, f0, iblock, dt):
    return os.path.join(snapDir,
        f'snp_nx{model.grid.shape[0]}_nz{model.grid.shape[1]}'
        f'_{model.grid.spacing[0]}_{model.grid.spacing[1]}'
        f'_FreqHz{f0 * 1000}_T{iblock * dt:05.2f}.zfp')



# 
# forward operator 

def ForwardOperator(model, geometry, kernel='sls'):

    m = model.m
    b = model.b
    qp = model.qp
    damp = model.damp
    rho = 1. / b
    f0 = geometry._f0
    dt = model.critical_dt

    p = TimeFunction(name='p', grid=model.grid, time_order=2,
                     space_order=SO, staggered=NODE)

    s = model.grid.stepping_dim.spacing
    t0 = p.indices[0] - s / 2

    extra_eqs = []
    r = None

    if kernel == 'sls':
        t_s = (sp.sqrt(1. + 1./qp**2) - 1./qp) / f0
        t_ep = 1. / (f0**2 * t_s)
        tt = (t_ep / t_s) - 1.

        r = TimeFunction(name='r', grid=model.grid, time_order=2,
                         space_order=SO, staggered=NODE)

        # Attenuation memory variable
        pde_r = r.dt - (tt / t_s) * rho * \
            div(b * grad(p, shift=.5), shift=-.5) + (1. / t_s) * r
        u_r = Eq(r.forward, damp * solve(pde_r, r.forward))

        # Pressure
        pde_p = m * p.dt2 - rho * (1. + tt) * \
            div(b * grad(p, shift=.5), shift=-.5) + \
            r.forward + (1 - damp) * p.dt
        u_p = Eq(p.forward, damp * solve(pde_p, p.forward))

        extra_eqs = [u_r]
        stencil = u_p

    elif kernel == 'kv':
        w0 = 2. * np.pi * f0
        tau = 1 / (w0 * qp)

        pde_p = m * p.dt2 - rho * \
            div(b * grad(p, shift=.5), shift=-.5) - \
            tau * rho * div(b * grad(p.dt(x0=t0), shift=.5), shift=-.5) + \
            (1 - damp) * p.dt
        stencil = Eq(p.forward, solve(pde_p, p.forward))

    elif kernel == 'maxwell':
        w0 = 2. * np.pi * f0

        pde_p = m * p.dt2 - rho * \
            div(b * grad(p, shift=.5), shift=-.5) + \
            m * w0 / qp * p.dt(x0=t0) + (1 - damp) * p.dt
        stencil = Eq(p.forward, solve(pde_p, p.forward))

    # Source injection
    src = geometry.src
    rec = geometry.rec
    dt_sym = model.grid.time_dim.spacing
    scale = dt_sym**2 / m

    src_term = src.inject(field=p.forward, expr=src * scale)
    rec_term = rec.interpolate(expr=p)

    op = Operator(extra_eqs + [stencil] + src_term + rec_term,
                  subs=model.spacing_map, name=f'Forward_{kernel}')

    return op, p, rec, r



def ImagingOperator(model, geometry, image, kernel='sls'):
    """
    Symbolic adjoint viscoacoustic operator — matches operators.py
    2nd order adjoint equations for SLS, KV, Maxwell.
    Includes imaging condition (cross-correlation).
    """
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=SO)
    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=SO,
                     staggered=NODE)

    m = model.m
    b = model.b
    qp = model.qp
    damp = model.damp
    rho = 1. / b
    f0 = geometry._f0
    dt = model.critical_dt

    s = model.grid.stepping_dim.spacing
    t0 = v.indices[0] - s / 2

    extra_eqs = []

    if kernel == 'sls':
        t_s = (sp.sqrt(1. + 1./qp**2) - 1./qp) / f0
        t_ep = 1. / (f0**2 * t_s)
        tt = (t_ep / t_s) - 1.

        r = TimeFunction(name='r', grid=model.grid, time_order=2,
                         space_order=SO, staggered=NODE)

        pde_r = r.dt.T + (tt / t_s) * v + (1. / t_s) * r
        u_r = Eq(r.backward, damp * solve(pde_r, r.backward))

        pde_v = m * v.dt2 - \
            div(b * grad((1. + tt) * rho * v, shift=.5), shift=-.5) - \
            div(b * grad(rho * r.backward, shift=.5), shift=-.5) + \
            (1 - damp) * v.dt.T
        stencil = Eq(v.backward, solve(pde_v, v.backward))
        extra_eqs = [u_r]

    elif kernel == 'kv':
        w0 = 2. * np.pi * f0
        tau = 1 / (w0 * qp)

        pde_v = m * v.dt2 - \
            div(b * grad(rho * v, shift=.5), shift=-.5) - \
            div(b * grad(rho * tau * v.dt(x0=t0).T, shift=.5), shift=-.5) + \
            (1 - damp) * v.dt.T
        stencil = Eq(v.backward, solve(pde_v, v.backward))

    elif kernel == 'maxwell':
        w0 = 2. * np.pi * f0

        pde_v = m * v.dt2 + m * w0 / qp * v.dt(x0=t0).T + \
            (1 - damp) * v.dt.T - \
            div(b * grad(rho * v, shift=.5), shift=-.5)
        stencil = Eq(v.backward, solve(pde_v, v.backward))

    # Residual injection at receiver locations
    residual = PointSource(name='residual', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / m)

    # Imaging condition: cross-correlate forward (u) and backward (v)
    image_update = Eq(image, image - u * v)

    op = Operator(extra_eqs + [stencil] + res_term + [image_update],
                  subs=model.spacing_map, name=f'Imaging_{kernel}')

    return op, u, v



# RTM loop

kernels = ['sls']  # Add 'kv', 'maxwell' as needed
results = {}

for kernel in kernels:
    print(f"\n{'='*60}")
    print(f"  RTM with {kernel.upper()} kernel")
    print(f"{'='*60}")

    # Build symbolic forward operator
    op_fwd, p_fwd, rec_fwd, r_fwd = ForwardOperator(model, geometry, kernel=kernel)
    op_fwd_smooth, p_smooth, rec_smooth, r_smooth = ForwardOperator(model0, geometry, kernel=kernel)

    # Build symbolic imaging operator
    image = Function(name='image', grid=model.grid)
    op_img, u_img, v_img = ImagingOperator(model, geometry, image, kernel=kernel)

    # Also run a full forward for the seismogram comparison
    solver = ViscoacousticWaveSolver(model, geometry, space_order=SO,
                                     kernel=kernel, time_order=2)
    geometry.src_positions[0, :] = source_locations[0, :]
    true_d_full, _, _, _ = solver.forward(vp=model.vp)
    results[kernel] = {'rec': np.array(true_d_full.data).copy()}

    # RTM per shot
    for i in range(nshots):
        logging.info(f'Imaging source {i+1} out of {nshots}')
        geometry.src_positions[0, :] = source_locations[i, :]

        # Snapshot directory
        snapDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f'snaps/{kernel}/shot{i:04d}')
        os.makedirs(snapDir, exist_ok=True)

        timeblocks_start = list(range(0, geometry.nt - timesteps_IC - 1, timesteps_IC))

        
        # FORWARD: true model (for observed data)
        
        print(f"  Forward (true model) — full run")
        p_fwd.data[:] = 0.
        if r_fwd is not None:
            r_fwd.data[:] = 0.
        rec_fwd.data[:] = 0.

        if kernel == 'sls':
            op_fwd.apply(p=p_fwd, r=r_fwd, rec=rec_fwd,
                         src=geometry.src, dt=model.critical_dt,
                         vp=model.vp, b=model.b, qp=model.qp)
        else:
            op_fwd.apply(p=p_fwd, rec=rec_fwd,
                         src=geometry.src, dt=model.critical_dt,
                         vp=model.vp, b=model.b, qp=model.qp)

        true_data = np.array(rec_fwd.data).copy()

        
        # FORWARD: smooth model (time-blocked + ZFP)
        
        print(f"  Forward (smooth model) — time-blocked with snapshots")
        p_smooth.data[:] = 0.
        if r_smooth is not None:
            r_smooth.data[:] = 0.
        rec_smooth.data[:] = 0.

        for iblock in timeblocks_start:
            time_m = iblock
            time_M = iblock + timesteps_IC

            print(f"    Forward block: {time_m} -> {time_M}")

            if kernel == 'sls':
                op_fwd_smooth.apply(p=p_smooth, r=r_smooth, rec=rec_smooth,
                                     src=geometry.src, dt=model0.critical_dt,
                                     vp=model0.vp, b=model0.b, qp=model0.qp,
                                     time_m=time_m, time_M=time_M)
            else:
                op_fwd_smooth.apply(p=p_smooth, rec=rec_smooth,
                                     src=geometry.src, dt=model0.critical_dt,
                                     vp=model0.vp, b=model0.b, qp=model0.qp,
                                     time_m=time_m, time_M=time_M)

            # Save snapshot with ZFP compression
            snpFn = snap_filename(snapDir, model, f0, iblock, geometry.dt)
            save_snapshot(np.array(p_smooth.data[0, SO:-SO, SO:-SO]), snpFn)
            print(f"    Saved {snpFn}")

        smooth_data = np.array(rec_smooth.data).copy()

    
        # Compute residual + mute
        
        residual = smooth_data - true_data
        residual[:mute_samples, :] = 0.

        
        # BACKWARD + IMAGING
        
        print(f"  Backward (imaging) — time-blocked with loaded snapshots")

        for iblock in reversed(timeblocks_start):
            time_m = iblock - timesteps_IC
            time_M = iblock

            if time_m < 0:
                continue

            

            # Load forward snapshot into u
            snpFn = snap_filename(snapDir, model, f0, iblock, geometry.dt)
            u_img.data[0, SO:-SO, SO:-SO] = load_snapshot(snpFn)
            

            # Run backward + imaging condition
            op_img.apply(u=u_img, vp=model0.vp, b=model0.b, qp=model0.qp,
                         dt=model0.critical_dt, residual=residual,
                         time_m=time_m, time_M=time_M)

    # ============================================
    # Plot results
    # ============================================
    img_diff = np.diff(np.array(image.data), axis=1)
    vmax_img = np.percentile(np.abs(img_diff), 99)
    if vmax_img == 0:
        vmax_img = 1.0
    results[kernel]['image_diff'] = img_diff

    plot_image(img_diff, vmin=-vmax_img, vmax=vmax_img)
    plt.title(f'{kernel.upper()} RTM Image')
    plt.show()

    print(f"  {kernel.upper()} image norm: {norm(image):.4f}")


# ==============================================================
# Compare kernels
# ==============================================================
print(f"\n{'='*60}")
print("  Comparing kernels")
print(f"{'='*60}")

# Seismograms
fig, axes = plt.subplots(1, len(kernels), figsize=(7*len(kernels), 8), sharey=True)
if len(kernels) == 1:
    axes = [axes]
for ax, kernel in zip(axes, kernels):
    rec = results[kernel]['rec']
    vmax = np.percentile(np.abs(rec), 99)
    if vmax == 0:
        vmax = 1.0
    ax.imshow(rec, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax,
              extent=[0, model.domain_size[0], tn, t0])
    ax.set_title(f'{kernel.upper()}', fontsize=14)
    ax.set_xlabel('X (m)')
axes[0].set_ylabel('Time (ms)')
fig.suptitle('Seismogram Comparison', fontsize=16)
plt.tight_layout()
plt.show()

# RTM images
fig, axes = plt.subplots(1, len(kernels), figsize=(7*len(kernels), 6), sharey=True)
if len(kernels) == 1:
    axes = [axes]
for ax, kernel in zip(axes, kernels):
    img = results[kernel]['image_diff']
    vmax = np.percentile(np.abs(img), 99)
    if vmax == 0:
        vmax = 1.0
    ax.imshow(img.T, aspect='auto', cmap='gray', vmin=-vmax, vmax=vmax)
    ax.set_title(f'{kernel.upper()}', fontsize=14)
    ax.set_xlabel('X (gridpoints)')
axes[0].set_ylabel('Depth (gridpoints)')
fig.suptitle('RTM Image Comparison', fontsize=16)
plt.tight_layout()
plt.show()

# Summary
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
for kernel in kernels:
    rec_norm = np.linalg.norm(results[kernel]['rec'])
    img_norm = np.linalg.norm(results[kernel]['image_diff'])
    print(f"  {kernel.upper():>8s} | rec norm: {rec_norm:.3f} | image norm: {img_norm:.4f}")