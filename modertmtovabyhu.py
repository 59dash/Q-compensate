import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

from devito import configuration
configuration['log-level'] = 'WARNING'

from devito import (TimeFunction, Function, Operator, Eq, solve,
                    norm, gaussian_smooth, div, grad, NODE)
from examples.seismic import (demo_model, AcquisitionGeometry, PointSource,
                               plot_image, plot_velocity)
from examples.seismic.viscoacoustic import ViscoacousticWaveSolver


# Model

def create_model(grid=None):
    model = demo_model('layers-viscoacoustic', origin=(0., 0.), shape=(101, 101),
                        spacing=(10., 10.), nbl=20, grid=grid, nlayers=2)
    return model

filter_sigma = (1, 1)
nshots = 21
nreceivers = 101
t0 = 0.
tn = 1000.
f0 = 0.010

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
source_locations[:, 0] = np.linspace(0., 1000, num=nshots)
source_locations[:, 1] = 30.

mute_samples = int(200 / model0.critical_dt)



# Viscoacoustic ImagingOperator — adjoint per kernel

def ImagingOperator(model, geometry, image, kernel='sls'):
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=4,
                     save=geometry.nt)

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
                         space_order=4, staggered=NODE)

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

    residual = PointSource(name='residual', grid=model.grid,
                           time_range=geometry.time_axis,
                           coordinates=geometry.rec_positions)
    res_term = residual.inject(field=v.backward, expr=residual * dt**2 / m)

    image_update = Eq(image, image - u * v)

    return Operator(extra_eqs + [stencil] + res_term + [image_update],
                    subs=model.spacing_map)



# RTM loop for each kernel

kernels = ['sls', 'kv', 'maxwell']
results = {}

for kernel in kernels:
    print(f"\n{'='*60}")
    print(f"  RTM with {kernel.upper()} kernel")
    print(f"{'='*60}")

    # Forward solver
    solver = ViscoacousticWaveSolver(model, geometry, space_order=4,
                                     kernel=kernel, time_order=2)

    # Single shot seismogram for comparison
    geometry.src_positions[0, :] = source_locations[nshots // 2, :]
    true_d, _, _, _ = solver.forward(vp=model.vp)
    results[kernel] = {'rec': np.array(true_d.data).copy()}

    # Image + viscoacoustic imaging operator
    image = Function(name='image', grid=model.grid)
    op_imaging = ImagingOperator(model, geometry, image, kernel=kernel)

    #  RTM
    for i in range(nshots):
        print('Imaging source %d out of %d' % (i+1, nshots))

        geometry.src_positions[0, :] = source_locations[i, :]

        true_d, _, _, _ = solver.forward(vp=model.vp)
        smooth_d, u0, _, _ = solver.forward(vp=model0.vp, save=True)

        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=4)
        residual = smooth_d.data - true_d.data
        residual[:mute_samples, :] = 0.

        op_imaging(u=u0, v=v, vp=model0.vp, dt=model0.critical_dt,
                   residual=residual)

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
    ax.plot(trace, t_axis, color=colors[kernel], label=kernel.upper(), linewidth=0.8)
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
    print(f"  {kernel.upper():>8s} | rec norm: {rec_norm:.3f} | image norm: {img_norm:.4f}")