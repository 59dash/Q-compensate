import sys, os
import logging

# use the devitoPro
devitopro_dir = "/home/xlz/Python/devitopro/devitopro-trial-ou"
devito_dir = devitopro_dir + "/submodules/devito"
logging.info(f" devito_dir is in : {devito_dir}")
sys.path.insert(0, devito_dir)
sys.path.insert(0, devitopro_dir)


from examples.seismic.model import SeismicModel
from examples.seismic import (AcquisitionGeometry, PointSource,
                              plot_image, plot_velocity, demo_model)
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver
from devito import (TimeFunction, Function, Operator, Eq, solve,
                    norm, gaussian_smooth, div, grad, NODE)
from devito import configuration
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from agl_model import agl_model

import zfpy # for compression

from contextlib import suppress

with suppress(ImportError):
    import pytest

from devito.logger import info
from examples.seismic import seismic_args, setup_geometry

def read_and_decompress_zfp(file_path):
    with open(file_path, 'rb') as f:
        decompressed_data = zfpy.decompress_numpy(f.read())
    return decompressed_data

# configuration['log-level'] = 'WARNING'
configuration['log-level'] = 'DEBUG'

# from examples.seismic import (demo_model, AcquisitionGeometry, PointSource,
# plot_image, plot_velocity)


# flag_model = 'demo'  # 'demo' or 'marmousi'
flag_model = 'marmousi'

# Model
match flag_model:
    case 'demo':
        def create_model(grid=None):
            model = demo_model('layers-viscoacoustic', origin=(0., 0.), shape=(101, 101),
                               spacing=(10., 10.), nbl=20, grid=grid, nlayers=2)
            return model
    case 'marmousi':
        def create_model(grid=None):
            model = agl_model('marmousi-agl-vp', data_path="/home/xlz/marmousi",
                              #spacing=(200., 200.),
                              grid=grid, 
                              nbl=100)
            return model


filter_sigma = (1, 1)
nshots = 1
nreceivers = 1701
t0 = 0.
tn = 4000.
f0 = 0.010
SO = 4  # space order

model = create_model()
model0 = create_model(grid=model.grid)
gaussian_smooth(model0.vp, sigma=filter_sigma)

# Geometry
src_coordinates = np.empty((1, 2))
src_coordinates[0, :] = np.array(model.domain_size) * .5
src_coordinates[0, -1] = 20.

rec_coordinates = np.empty((nreceivers, 2))
rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
rec_coordinates[:, 1] = 30.

geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                               t0, tn, f0=f0, src_type='Ricker')

source_locations = np.empty((nshots, 2), dtype=np.float32)
# source_locations[:, 0] = np.linspace(0., 1000, num=nshots)
# source_locations[:, 1] = 30.
source_locations[0, 0] = model.domain_size[0] * 0.5
print(f" source_locations is in : {source_locations}")
mute_samples = int(200 / model0.critical_dt)
def ForwardOperator(model, geometry, kernel='sls'):
    m = model.m
    b = model.b
    qp = model.qp
    damp = model.damp
    rho = 1. / b
    f0 = geometry._f0
    dt = model.critical_dt

    p = p or TimeFunction(name="p", grid=model.grid,
                              time_order=2, space_order=SO,
                              staggered=NODE)
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
    
    src = geometry.src
    rec = geometry.rec
    dt_sym = model.grid.time_dim.spacing
    scale = dt_sym**2 / m
 
    src_term = src.inject(field=p.forward, expr=src * scale)
    rec_term = rec.interpolate(expr=p)
 
    op = Operator(extra_eqs + [stencil] + src_term + rec_term,
                  subs=model.spacing_map, name=f'Forward_{kernel}')
 
    return op, p, rec, r
# Viscoacoustic ImagingOperator — adjoint per kernel

def ImagingOperator(model, geometry, image, kernel='sls'):
    v = TimeFunction(name='v', grid=model.grid, time_order=2,
                         space_order=SO)
    
    # u is the forward wavefield, which we need to save for the imaging condition. We can save it in full, or use compression to reduce memory usage.
    
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
    # t0 = v.indices[0] - s / 2

    extra_eqs = []

    if kernel == 'sls':
        t_s = (sp.sqrt(1. + 1./qp**2) - 1./qp) / f0
        t_ep = 1. / (f0**2 * t_s)
        tt = (t_ep / t_s) - 1.

        r = TimeFunction(name='r', grid=model.grid, time_order=2,
                         space_order=SO, staggered=NODE)

        pde_r = r.dt.T + (tt / t_s) * v + (1. / t_s) * r
        # pde_r = r.dt.T + (tt / t_s) * v + (1. / t_s) * r
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

    residual = PointSource(name='residual', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / m)
    
    image_update = Eq(image, image - u * v)
    
    return Operator(extra_eqs + [stencil] + res_term + [image_update],
                    subs=model.spacing_map)


# RTM loop for each kernel
# kernels = ['sls', 'kv', 'maxwell']
kernels = ['sls']
results = {}

for kernel in kernels:
    print(f"\n{'='*60}")
    print(f"  RTM with {kernel.upper()} kernel")
    print(f"{'='*60}")

    # Forward solver
    solver = ViscoacousticWaveSolver(model, geometry, space_order=SO,
                                     kernel=kernel, time_order=2, 
                                     )

    # Single shot seismogram for comparison
    geometry.src_positions[0, :] = source_locations[nshots // 2, :]
    true_d, _, _, _ = solver.forward(vp=model.vp)
    results[kernel] = {'rec': np.array(true_d.data).copy()}

    # Image + viscoacoustic imaging operator
    image = Function(name='image', grid=model.grid)
    op_imaging = ImagingOperator(model, geometry, image, kernel=kernel)

    #  RTM
    for i in range(nshots):
        logging.info('Imaging source %d out of %d' % (i+1, nshots))

        geometry.src_positions[0, :] = source_locations[i, :]

        true_d, _, _, _ = solver.forward(vp=model.vp)

        # # update the snapshots by time block and save the snapshots in disk with compression

        # timesteps_IC = int( 1./ (geometry.f0 * 4) / geometry.dt )  # time step for saving the snapshots for imaging condition, which is usually larger than the simulation time step to save memory and speed up the imaging condition. Here we use 1/4 of the period of the source wavelet as a rule of thumb.
        timesteps_IC = 4 # time step for saving the snapshots for imaging condition, which is usually larger than the simulation time step to save memory and speed up the imaging condition. Here we use 1/4 of the period of the source wavelet as a rule of thumb.

        ## create a directory to save the snapshots for imaging condition
        snapDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'snaps/{kernel}/shot{i:04d}')
        if not os.path.exists(snapDir):
            os.makedirs(snapDir)

        timeblocks_start = range(0, geometry.nt - timesteps_IC - 1, timesteps_IC)

        for iblock in timeblocks_start:

            print(f" Forward: time block {iblock} to {iblock+timesteps_IC}")            

            time_m = iblock # the starting time step for the forward simulation to save the snapshots for imaging condition
            time_M = iblock + timesteps_IC# the ending time step for the forward simulation to save the snapshots for imaging condition

            smooth_d, u0, _, _ = solver.forward(vp=model0.vp, save=False,
                                                time_m = time_m, time_M = time_M)

            snpFn = (f'snp_nx{model.grid.shape[0]}_nz{model.grid.shape[1]}_{model.grid.spacing[0]}_{model.grid.spacing[1]}'
                        f'_FreqHz{f0 * 1000}_T{iblock*geometry.dt:05.2f}.zfp')

            snpFn = os.path.join(snapDir, snpFn)
            with open(snpFn, 'wb') as f:
                f.write(zfpy.compress_numpy(u0.data[0,:,:], precision=16))
            print(f'Save snapshot for shot {i} in {snpFn}')

        residual = smooth_d.data - true_d.data
        residual[:mute_samples, :] = 0.

        # backward propagation and imaging condition
        for iblock in reversed(timeblocks_start):

            print(f" Backward: time block {iblock} to {iblock+timesteps_IC}")            

            time_M = iblock # the starting time step for the forward simulation to save the snapshots for imaging condition
            time_m = iblock - timesteps_IC# the ending time step for the forward simulation to save the snapshots for imaging condition
            if time_m < 0:
                continue

            # load the snapshots for imaging condition
            snpFn = (f'snp_nx{model.grid.shape[0]}_nz{model.grid.shape[1]}_{model.grid.spacing[0]}_{model.grid.spacing[1]}'
                        f'_FreqHz{f0 * 1000}_T{iblock*geometry.dt:05.2f}.zfp')

            snpFn = os.path.join(snapDir, snpFn)
            u0._data[0,SO:-SO,SO:-SO] = read_and_decompress_zfp(snpFn)
            print(f'load snapshot for shot {i} in {snpFn}')

            # backward propagation and imaging condition with the loaded snapshots for imaging condition
            op_imaging.apply(u=u0, vp=model0.vp, dt=model0.critical_dt,
                    residual=residual, 
                    time_m = time_m, time_M = time_M)
        
        ######################################################################### compression done###########################################################################################
    # Store and plot
    img_diff = np.diff(np.array(image.data), axis=1)
    vmax_img = np.percentile(np.abs(img_diff), 99)
    if vmax_img == 0:
        vmax_img = 1.0
    results[kernel]['image_diff'] = img_diff

    plot_image(img_diff, vmin=-vmax_img, vmax=vmax_img)
    plt.title(f'{kernel.upper()} RTM Image')
    plt.show()

    print(f"  {kernel.upper()} image norm: {norm(image):.4f}")


# Compare the 3 kernels

print(f"\n{'='*60}")
print("  Comparing SLS vs KV vs Maxwell")
print(f"{'='*60}")

# Seismograms side by side
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
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



# Seismogram differences
pairs = [('sls', 'kv'), ('sls', 'maxwell'), ('kv', 'maxwell')]
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
for ax, (k1, k2) in zip(axes, pairs):
    diff = results[k1]['rec'] - results[k2]['rec']
    vmax = np.percentile(np.abs(diff), 99)
    if vmax == 0:
        vmax = 1.0
    ax.imshow(diff, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax,
              extent=[0, model.domain_size[0], tn, t0])
    ax.set_title(f'{k1.upper()} - {k2.upper()}', fontsize=14)
    ax.set_xlabel('X (m)')
axes[0].set_ylabel('Time (ms)')
fig.suptitle('Seismogram Differences', fontsize=16)
plt.tight_layout()
plt.show()

# RTM images side by side
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
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

# RTM image differences
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
for ax, (k1, k2) in zip(axes, pairs):
    diff = results[k1]['image_diff'] - results[k2]['image_diff']
    vmax = np.percentile(np.abs(diff), 99)
    if vmax == 0:
        vmax = 1.0
    ax.imshow(diff.T, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
    ax.set_title(f'{k1.upper()} - {k2.upper()}', fontsize=14)
    ax.set_xlabel('X (gridpoints)')
axes[0].set_ylabel('Depth (gridpoints)')
fig.suptitle('RTM Image Differences', fontsize=16)
plt.tight_layout()
plt.show()

# Trace comparison
fig, ax = plt.subplots(figsize=(6, 10))
colors = {'sls': 'blue', 'kv': 'red', 'maxwell': 'green'}
for kernel in kernels:
    rec = results[kernel]['rec']
    mid = rec.shape[1] // 2
    trace = rec[:, mid]
    t_axis = np.linspace(t0, tn, len(trace))
    ax.plot(trace, t_axis, color=colors[kernel],
            label=kernel.upper(), linewidth=0.8)
ax.invert_yaxis()
ax.set_xlabel('Amplitude')
ax.set_ylabel('Time (ms)')
ax.set_title('Trace Comparison (Middle Receiver)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Summary
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
for kernel in kernels:
    rec_norm = np.linalg.norm(results[kernel]['rec'])
    img_norm = np.linalg.norm(results[kernel]['image_diff'])
    print(
        f"  {kernel.upper():>8s} | rec norm: {rec_norm:.3f} | image norm: {img_norm:.4f}")