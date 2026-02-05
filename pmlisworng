#the PML is wrong it is absorb rather than PML
"""
Q-compensated Reverse-Time Migration (Q-RTM)
Based on: Zhu et al. (2014) - Q-compensated reverse-time migration

This implementation includes:
1. Viscoacoustic wave equation with separated attenuation and dispersion
2. Forward and backward wavefield extrapolation with compensation
3. Zero-lag crosscorrelation imaging condition
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
#from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.ndimage import gaussian_filter
from q_compensated_rtm_re import QRTM, create_two_layer_model

def tukey_window(N, alpha=0.5):
    """
    Generate Tukey window (tapered cosine window)
    
    Parameters:
    -----------
    N : int
        Number of points
    alpha : float
        Taper ratio (0 = rectangular, 1 = Hann)
    """
    n = np.arange(N)
    width = int(np.floor(alpha*(N-1)/2.0))
    n1 = n[:width+1]
    n2 = n[width+1:N-width-1]
    n3 = n[N-width-1:]
    
    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0*n1/alpha/(N-1))))
    w2 = np.ones(len(n2))
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(N-1))))
    
    w = np.concatenate((w1, w2, w3))
    return w


class QRTM:
    """Q-compensated Reverse-Time Migration implementation"""
    
    def __init__(self, nx: int, nz: int, dx: float, dz: float, dt: float, 
                 npml: int = 30):
        """
        Initialize Q-RTM
        
        Parameters:
        -----------
        nx : int
            Number of grid points in x-direction
        nz : int
            Number of grid points in z-direction
        dx : float
            Grid spacing in x-direction (meters)
        dz : float
            Grid spacing in z-direction (meters)
        dt : float
            Time step (seconds)
        npml : int
            Number of PML boundary points (default: 20)
        """
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.npml = npml
        
        # Wavenumber grids for spectral operations
        self.kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
        self.kz = 2 * np.pi * np.fft.fftfreq(nz, dz)
        self.KX, self.KZ = np.meshgrid(self.kx, self.kz)
        self.K2 = self.KX**2 + self.KZ**2  # Laplacian in wavenumber domain
        
        # Create PML damping profile
        self.pml_damping = self._create_pml_damping()
        
    def compute_q_parameters(self, Q: float, f0: float, c0: float) -> Tuple[float, float, float]:
        """
        Compute Q-related parameters
        
        Parameters:
        -----------
        Q : float
            Quality factor
        f0 : float
            Reference frequency (Hz)
        c0 : float
            Reference velocity (m/s)
            
        Returns:
        --------
        gamma : float
            Attenuation parameter
        tau : float
            Attenuation coefficient
        eta : float
            Dispersion coefficient
        """
        # gamma = (1/pi) * arctan(1/Q)
        gamma = (1/np.pi) * np.arctan(1/Q)
        
        # Reference angular frequency
        w0 = 2 * np.pi * f0
        
        # Attenuation coefficient: tau = -c0^(2*gamma-1) * w0^(-2*gamma) * sin(pi*gamma)
        tau = -c0**(2*gamma - 1) * w0**(-2*gamma) * np.sin(np.pi * gamma)
        
        # Dispersion coefficient: eta = -c0^(2*gamma) * w0^(-2*gamma) * cos(pi*gamma)
        eta = -c0**(2*gamma) * w0**(-2*gamma) * np.cos(np.pi * gamma)
        
        return gamma, tau, eta
    
    def _create_pml_damping(self) -> np.ndarray:
        """
        Create PML (Perfectly Matched Layer) damping profile
        
        Returns:
        --------
        damping : np.ndarray
            Damping coefficient array (nz, nx)
        """
        
        damping = np.zeros((self.nz, self.nx))
        
        # PML parameters
        R = 0.001  # Reflection coefficient (0.1% reflection)
        d0 = -4.0 * np.log(R) / (2.0 * self.npml)
        
        # Create damping profile for each boundary
        for i in range(self.nz):
            for j in range(self.nx):
                dist_z = 0.0
                dist_x = 0.0
                
                # Top boundary
                if i < self.npml:
                    dist_z = (self.npml - i) / self.npml
                
                # Bottom boundary
                if i >= self.nz - self.npml:
                    dist_z = (i - (self.nz - self.npml - 1)) / self.npml
                
                # Left boundary
                if j < self.npml:
                    dist_x = (self.npml - j) / self.npml
                
                # Right boundary
                if j >= self.nx - self.npml:
                    dist_x = (j - (self.nx - self.npml - 1)) / self.npml
                
                # Combine damping (quadratic profile)
                damping[i, j] = d0 * (dist_z**2 + dist_x**2)
        
        return damping
    
    def apply_pml(self, p: np.ndarray) -> np.ndarray:
        """
        Apply PML damping to wavefield
        
        Parameters:
        -----------
        p : np.ndarray
            Pressure field
            
        Returns:
        --------
        np.ndarray
            Damped pressure field
        """
        # Exponential damping
        return p * np.exp(-self.pml_damping * self.dt)
    
    def fractional_laplacian(self, p: np.ndarray, gamma: float, order: float = 1.0) -> np.ndarray:
        """
        Compute fractional Laplacian: (-∇²)^(gamma + order)
        
        Parameters:
        -----------
        p : np.ndarray
            Pressure field
        gamma : float
            Attenuation parameter
        order : float
            Order offset (default 1.0)
            
        Returns:
        --------
        np.ndarray
            Fractional Laplacian of p
        """
        # Transform to wavenumber domain
        p_k = np.fft.fft2(p)
        
        # Apply fractional Laplacian: (k²)^(gamma + order)
        power = gamma + order
        result_k = p_k * (self.K2 + 1e-10)**power  # Add small epsilon to avoid division by zero
        
        # Transform back to spatial domain
        result = np.real(np.fft.ifft2(result_k))
        
        return result
    
    def apply_lowpass_filter(self, p_k: np.ndarray, f_cutoff: float, 
                            c_max: float, taper_ratio: float = 0.5) -> np.ndarray:
        """
        Apply Tukey low-pass filter to prevent high-frequency amplification
        
        Parameters:
        -----------
        p_k : np.ndarray
            Pressure field in wavenumber domain
        f_cutoff : float
            Cutoff frequency (Hz)
        c_max : float
            Maximum velocity (m/s)
        taper_ratio : float
            Taper ratio for Tukey window
            
        Returns:
        --------
        np.ndarray
            Filtered field in wavenumber domain
        """
        # Cutoff wavenumber
        k_cutoff = 2 * np.pi * f_cutoff / c_max
        
        # Magnitude of wavenumber
        K_mag = np.sqrt(self.K2)
        
        # Create Tukey window in wavenumber domain
        filter_1d = np.ones_like(K_mag.flatten())
        k_flat = K_mag.flatten()
        
        # Apply taper
        n_points = len(filter_1d)
        n_taper = int(taper_ratio * n_points)
        
        for i in range(n_points):
            if k_flat[i] > k_cutoff * (1 - taper_ratio):
                if k_flat[i] < k_cutoff:
                    # Taper region
                    alpha = (k_flat[i] - k_cutoff * (1 - taper_ratio)) / (k_cutoff * taper_ratio)
                    filter_1d[i] = 0.5 * (1 + np.cos(np.pi * alpha))
                else:
                    # Stop band
                    filter_1d[i] = 0.0
        
        filter_2d = filter_1d.reshape(K_mag.shape)
        
        return p_k * filter_2d
    
    def viscoacoustic_step(self, p: np.ndarray, p_old: np.ndarray, 
                          c0: np.ndarray, Q: np.ndarray, f0: float,
                          compensate: bool = False, 
                          f_cutoff: Optional[float] = None) -> np.ndarray:
        """
        One time step of viscoacoustic wave equation
        
        Equation 9 (forward) or Equation 10 (compensated):
        (1/c0²) * ∂²p/∂t² = ∇²p + β1*{η*L - ∇²}p + β2*τ*∂/∂t*H*p
        
        where:
        - L = (-∇²)^(γ+1) : dispersion operator
        - H = (-∇²)^(γ+1/2) : attenuation operator
        - β1 = 1 (always for dispersion)
        - β2 = +1 (forward), -1 (compensated)
        
        Parameters:
        -----------
        p : np.ndarray
            Current pressure field
        p_old : np.ndarray
            Previous pressure field
        c0 : np.ndarray
            Reference velocity model
        Q : np.ndarray
            Quality factor model
        f0 : float
            Reference frequency
        compensate : bool
            If True, apply compensation (β2 = -1)
        f_cutoff : Optional[float]
            Cutoff frequency for low-pass filter
            
        Returns:
        --------
        np.ndarray
            Next pressure field
        """
        # Compute average Q parameters
        Q_mean = np.mean(Q)
        c0_mean = np.mean(c0)
        gamma, tau, eta = self.compute_q_parameters(Q_mean, f0, c0_mean)
        
        # Standard Laplacian in wavenumber domain
        p_k = np.fft.fft2(p)
        laplacian_p_k = -self.K2 * p_k
        laplacian_p = np.real(np.fft.ifft2(laplacian_p_k))
        
        # Dispersion operator: {η*L - ∇²}p
        L_p = self.fractional_laplacian(p, gamma, order=1.0)
        dispersion_term = eta * L_p - laplacian_p
        
        # Attenuation operator: τ*∂/∂t*H*p
        # Approximate time derivative: ∂p/∂t ≈ (p - p_old) / dt
        p_dot = (p - p_old) / self.dt
        H_p_dot = self.fractional_laplacian(p_dot, gamma, order=0.5)
        
        # β2 = -1 for compensation, +1 for forward
        beta2 = -1 if compensate else 1
        attenuation_term = beta2 * tau * H_p_dot
        
        # Apply low-pass filter if specified
        if f_cutoff is not None:
            c_max = np.max(c0)
            
            # Filter dispersion term
            disp_k = np.fft.fft2(dispersion_term)
            disp_k = self.apply_lowpass_filter(disp_k, f_cutoff, c_max)
            dispersion_term = np.real(np.fft.ifft2(disp_k))
            
            # Filter attenuation term
            atten_k = np.fft.fft2(attenuation_term)
            atten_k = self.apply_lowpass_filter(atten_k, f_cutoff, c_max)
            attenuation_term = np.real(np.fft.ifft2(atten_k))
        
        # Total right-hand side
        rhs = laplacian_p + dispersion_term + attenuation_term
        
        # Time stepping (second-order accurate)
        # p_new = 2*p - p_old + dt² * c0² * rhs
        p_new = 2*p - p_old + (self.dt**2) * (c0**2) * rhs
        
        # Apply PML damping to prevent wraparound
        p_new = self.apply_pml(p_new)
        
        return p_new
    
    def add_source(self, p: np.ndarray, src_pos: Tuple[int, int], 
                   amplitude: float) -> np.ndarray:
        """Add source to pressure field"""
        ix, iz = src_pos
        p[iz, ix] += amplitude
        return p
    
    def ricker_wavelet(self, f0: float, nt: int, t0: float = 0.04) -> np.ndarray:
        """
        Generate Ricker wavelet
        
        Parameters:
        -----------
        f0 : float
            Peak frequency (Hz)
        nt : int
            Number of time samples
        t0 : float
            Onset time (seconds)
            
        Returns:
        --------
        np.ndarray
            Ricker wavelet
        """
        t = np.arange(nt) * self.dt
        return (1 - 2*(np.pi*f0*(t-t0))**2) * np.exp(-(np.pi*f0*(t-t0))**2)
    
    def forward_propagate(self, src_wavelet: np.ndarray, src_pos: Tuple[int, int],
                         c0: np.ndarray, Q: np.ndarray, f0: float,
                         compensate: bool = False,
                         f_cutoff: Optional[float] = None,
                         save_snapshots: bool = False) -> Tuple:
        """
        Forward propagate source wavefield
        
        Parameters:
        -----------
        src_wavelet : np.ndarray
            Source time function
        src_pos : Tuple[int, int]
            Source position (ix, iz)
        c0 : np.ndarray
            Velocity model
        Q : np.ndarray
            Q model
        f0 : float
            Reference frequency
        compensate : bool
            Apply Q-compensation
        f_cutoff : Optional[float]
            Cutoff frequency for filter
        save_snapshots : bool
            Save wavefield snapshots
            
        Returns:
        --------
        wavefields : np.ndarray or None
            Saved wavefields if save_snapshots=True
        """
        nt = len(src_wavelet)
        
        # Initialize fields
        p = np.zeros((self.nz, self.nx))
        p_old = np.zeros((self.nz, self.nx))
        
        # Storage for snapshots
        if save_snapshots:
            wavefields = np.zeros((nt, self.nz, self.nx))
        else:
            wavefields = None
        
        # Time loop
        for it in range(nt):
            # Add source
            p = self.add_source(p, src_pos, src_wavelet[it])
            
            # Propagate
            p_new = self.viscoacoustic_step(p, p_old, c0, Q, f0, 
                                           compensate, f_cutoff)
            
            # Update
            p_old = p.copy()
            p = p_new.copy()
            
            # Save snapshot
            if save_snapshots:
                wavefields[it] = p.copy()
            
            if it % 100 == 0:
                print(f"Forward propagation: {it}/{nt}")
        
        return wavefields
    
    def backward_propagate(self, receiver_data: np.ndarray, 
                          receiver_pos: np.ndarray,
                          c0: np.ndarray, Q: np.ndarray, f0: float,
                          compensate: bool = False,
                          f_cutoff: Optional[float] = None,
                          save_snapshots: bool = False) -> Tuple:
        """
        Backward propagate receiver wavefield
        
        Parameters:
        -----------
        receiver_data : np.ndarray
            Recorded data (nt, nr)
        receiver_pos : np.ndarray
            Receiver positions (nr, 2) - (ix, iz) pairs
        c0 : np.ndarray
            Velocity model
        Q : np.ndarray
            Q model
        f0 : float
            Reference frequency
        compensate : bool
            Apply Q-compensation
        f_cutoff : Optional[float]
            Cutoff frequency for filter
        save_snapshots : bool
            Save wavefield snapshots
            
        Returns:
        --------
        wavefields : np.ndarray or None
            Saved wavefields if save_snapshots=True
        """
        nt, nr = receiver_data.shape
        
        # Flip data in time for backward propagation
        data_reversed = receiver_data[::-1, :]
        
        # Initialize fields
        p = np.zeros((self.nz, self.nx))
        p_old = np.zeros((self.nz, self.nx))
        
        # Storage for snapshots
        if save_snapshots:
            wavefields = np.zeros((nt, self.nz, self.nx))
        else:
            wavefields = None
        
        # Time loop
        for it in range(nt):
            # Inject receiver data
            for ir in range(nr):
                ix, iz = receiver_pos[ir]
                p[iz, ix] += data_reversed[it, ir]
            
            # Propagate
            p_new = self.viscoacoustic_step(p, p_old, c0, Q, f0, 
                                           compensate, f_cutoff)
            
            # Update
            p_old = p.copy()
            p = p_new.copy()
            
            # Save snapshot
            if save_snapshots:
                wavefields[it] = p.copy()
            
            if it % 100 == 0:
                print(f"Backward propagation: {it}/{nt}")
        
        return wavefields
    
    def imaging_condition(self, source_wavefields: np.ndarray,
                         receiver_wavefields: np.ndarray) -> np.ndarray:
        """
        Zero-lag crosscorrelation imaging condition
        
        I(x) = ∫ S(x,t) * R(x,t) dt
        
        Parameters:
        -----------
        source_wavefields : np.ndarray
            Source wavefields (nt, nz, nx)
        receiver_wavefields : np.ndarray
            Receiver wavefields (nt, nz, nx)
            
        Returns:
        --------
        image : np.ndarray
            Migrated image (nz, nx)
        """
        # Reverse receiver wavefields to align with source
        receiver_rev = receiver_wavefields[::-1, :, :]
        
        # Crosscorrelation and integration over time
        image = np.sum(source_wavefields * receiver_rev, axis=0)
        
        return image


def create_two_layer_model(nx: int, nz: int, dx: float, dz: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create two-layer velocity and Q model (similar to paper's example)
    
    Parameters:
    -----------
    nx, nz : int
        Model dimensions
    dx, dz : float
        Grid spacing
        
    Returns:
    --------
    velocity : np.ndarray
        Velocity model (nz, nx)
    Q : np.ndarray
        Q model (nz, nx)
    """
    velocity = np.zeros((nz, nx))
    Q = np.zeros((nz, nx))
    
    # Layer 1: z < 750m
    layer1_depth = int(750.0 / dz)
    velocity[:layer1_depth, :] = 2000.0  # m/s
    Q[:layer1_depth, :] = 50.0
    
    # Layer 2: z >= 750m
    velocity[layer1_depth:, :] = 2500.0  # m/s
    Q[layer1_depth:, :] = 25.0
    
    return velocity, Q


def main():
    """
    Main function to demonstrate Q-RTM
    """
    from scipy.ndimage import gaussian_filter
    
    print("=" * 60)
    print("Q-compensated Reverse-Time Migration")
    print("Based on: Zhu et al. (2014)")
    print("=" * 60)
    
    # Model parameters
    nx, nz = 150, 150  # Grid points
    dx, dz = 10.0, 10.0  # meters
    dt = 0.001  # seconds
    npml = 30  # PML boundary thickness
    
    # Create velocity and Q models
    print("\n1. Creating velocity and Q models...")
    velocity, Q = create_two_layer_model(nx, nz, dx, dz)
    
    # Smooth models for migration (as mentioned in paper)
    from scipy.ndimage import gaussian_filter
    velocity_smooth = gaussian_filter(velocity, sigma=2.0)
    Q_smooth = gaussian_filter(Q, sigma=2.0)
    
    # Source parameters
    f0 = 25.0  # Hz (peak frequency)
    nt = 800  # time steps
    t0 = 0.04  # onset time
    
    # Source and receiver positions
    src_x, src_z = nx // 2, 1  # Middle of model, near surface
    src_pos = (src_x, src_z)
    
    # Receiver array
    nr = nx - 10
    receiver_pos = np.zeros((nr, 2), dtype=int)
    receiver_pos[:, 0] = np.arange(5, nx - 5)  # x positions
    receiver_pos[:, 1] = 5  # z position (near surface)
    
    # Initialize Q-RTM
    print("\n2. Initializing Q-RTM...")
    qrtm = QRTM(nx, nz, dx, dz, dt, npml=npml)
    print(f"   PML boundary thickness: {npml} grid points ({npml*dx}m)")
    
    # Generate source wavelet
    src_wavelet = qrtm.ricker_wavelet(f0, nt, t0)
    
    # ========================================
    # FORWARD MODELING (Generate synthetic data)
    # ========================================
    print("\n3. Forward modeling to generate synthetic data...")
    print("   (Using viscoacoustic wave equation)")
    
    # Forward propagate without compensation to generate data
    wavefields_fwd = qrtm.forward_propagate(
        src_wavelet, src_pos, velocity, Q, f0,
        compensate=False, f_cutoff=None, save_snapshots=True
    )
    
    # Extract receiver data
    receiver_data = np.zeros((nt, nr))
    for it in range(nt):
        for ir in range(nr):
            ix, iz = receiver_pos[ir]
            receiver_data[it, ir] = wavefields_fwd[it, iz, ix]
    
    print(f"   Generated {nr} receiver traces")
    
    # ========================================
    # ACOUSTIC RTM (Reference - no compensation)
    # ========================================
    print("\n4. Running acoustic RTM (reference)...")
    
    # Use acoustic equation (Q = infinity) for reference
    Q_acoustic = np.ones_like(Q) * 10000
    
    source_wf_acoustic = qrtm.forward_propagate(
        src_wavelet, src_pos, velocity_smooth, Q_acoustic, f0,
        compensate=False, f_cutoff=None, save_snapshots=True
    )
    
    receiver_wf_acoustic = qrtm.backward_propagate(
        receiver_data, receiver_pos, velocity_smooth, Q_acoustic, f0,
        compensate=False, f_cutoff=None, save_snapshots=True
    )
    
    image_acoustic = qrtm.imaging_condition(source_wf_acoustic, receiver_wf_acoustic)
    
    # ========================================
    # Q-RTM (With compensation)
    # ========================================
    print("\n5. Running Q-RTM with compensation...")
    
    f_cutoff = 120.0  # Hz (as mentioned in paper)
    
    source_wf_qrtm = qrtm.forward_propagate(
        src_wavelet, src_pos, velocity_smooth, Q_smooth, f0,
        compensate=True, f_cutoff=f_cutoff, save_snapshots=True
    )
    
    receiver_wf_qrtm = qrtm.backward_propagate(
        receiver_data, receiver_pos, velocity_smooth, Q_smooth, f0,
        compensate=True, f_cutoff=f_cutoff, save_snapshots=True
    )
    
    image_qrtm = qrtm.imaging_condition(source_wf_qrtm, receiver_wf_qrtm)
  
    # ========================================
    # VISUALIZATION
    # ========================================
    print("\n6. Creating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Velocity model
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(velocity, aspect='auto', cmap='jet', extent=[0, nx*dx, nz*dz, 0])
    ax1.set_title('Velocity Model (m/s)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Depth (m)')
    plt.colorbar(im1, ax=ax1)
    
    # Q model
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(Q, aspect='auto', cmap='jet', extent=[0, nx*dx, nz*dz, 0])
    ax2.set_title('Q Model', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Depth (m)')
    plt.colorbar(im2, ax=ax2)
    
    # Source wavelet
    ax3 = plt.subplot(3, 3, 3)
    t = np.arange(nt) * dt
    ax3.plot(t, src_wavelet, 'b-', linewidth=1.5)
    ax3.set_title(f'Source Wavelet (f0={f0} Hz)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True, alpha=0.3)
    
    # Receiver data (shot gather)
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.imshow(receiver_data, aspect='auto', cmap='seismic', 
                     extent=[0, nr*dx, nt*dt, 0])
    ax4.set_title('Receiver Data (Shot Gather)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Receiver Position (m)')
    ax4.set_ylabel('Time (s)')
    plt.colorbar(im4, ax=ax4)
    
    # Snapshot at t=0.3s (source wavefield - acoustic)
    snapshot_time = int(0.3 / dt)
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.imshow(source_wf_acoustic[snapshot_time], aspect='auto', cmap='seismic',
                     extent=[0, nx*dx, nz*dz, 0])
    ax5.set_title('Source Wavefield (Acoustic, t=0.3s)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Distance (m)')
    ax5.set_ylabel('Depth (m)')
    plt.colorbar(im5, ax=ax5)
    
    # Snapshot at t=0.3s (receiver wavefield - acoustic)
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.imshow(receiver_wf_acoustic[snapshot_time], aspect='auto', cmap='seismic',
                     extent=[0, nx*dx, nz*dz, 0])
    ax6.set_title('Receiver Wavefield (Acoustic, t=0.3s)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Distance (m)')
    ax6.set_ylabel('Depth (m)')
    plt.colorbar(im6, ax=ax6)
    
    # Acoustic RTM image
    ax7 = plt.subplot(3, 3, 7)
    vmax = np.percentile(np.abs(image_acoustic), 98)
    im7 = ax7.imshow(image_acoustic, aspect='auto', cmap='seismic',
                     extent=[0, nx*dx, nz*dz, 0], vmin=-vmax, vmax=vmax)
    ax7.set_title('Acoustic RTM (Reference)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Distance (m)')
    ax7.set_ylabel('Depth (m)')
    plt.colorbar(im7, ax=ax7)
    
    # Q-RTM image
    ax8 = plt.subplot(3, 3, 8)
    vmax_qrtm = np.percentile(np.abs(image_qrtm), 98)
    im8 = ax8.imshow(image_qrtm, aspect='auto', cmap='seismic',
                     extent=[0, nx*dx, nz*dz, 0], vmin=-vmax_qrtm, vmax=vmax_qrtm)
    ax8.set_title('Q-RTM (Compensated)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Distance (m)')
    ax8.set_ylabel('Depth (m)')
    plt.colorbar(im8, ax=ax8)
    
    # Comparison trace
    ax9 = plt.subplot(3, 3, 9)
    trace_x = nx // 2
    ax9.plot(image_acoustic[:, trace_x], np.arange(nz)*dz, 'b-', 
             label='Acoustic RTM', linewidth=1.5)
    ax9.plot(image_qrtm[:, trace_x], np.arange(nz)*dz, 'r--', 
             label='Q-RTM', linewidth=1.5)
    ax9.set_title(f'Trace Comparison (x={trace_x*dx}m)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Amplitude')
    ax9.set_ylabel('Depth (m)')
    ax9.invert_yaxis()
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/xlz/qrtm_results.png', dpi=150, bbox_inches='tight')
    print("   Saved figure: qrtm_results.png")
    
    print("\n" + "=" * 60)
    print("Q-RTM implementation complete!")
    print("=" * 60)
    print("\nKey observations:")
    print("1. Q-RTM compensates for amplitude loss in attenuating media")
    print("2. Phase dispersion is corrected by keeping dispersion operator sign")
    print("3. Low-pass filter prevents high-frequency noise amplification")
    print("4. Resulting image has improved resolution below attenuating layers")
    
    return qrtm, image_acoustic, image_qrtm


if __name__ == "__main__":
    qrtm, img_acoustic, img_qrtm = main()
