import numpy as np
import matplotlib.pyplot as plt

def UPA_codebook_generator(Mx, My, Mz, over_sampling_x, over_sampling_y, over_sampling_z, ant_spacing):
    """
    Generate UPA codebook for beamforming

    Parameters:
    Mx, My, Mz: Number of antenna elements in each dimension
    over_sampling_x, over_sampling_y, over_sampling_z: Oversampling factors
    ant_spacing: Antenna spacing

    Returns:
    F_CB: Codebook matrix
    all_beams: Beam indices
    """

    kd = 2 * np.pi * ant_spacing
    antx_index = np.arange(Mx)
    anty_index = np.arange(My)
    antz_index = np.arange(Mz)
    M = Mx * My * Mz

    # Codebook sizes
    codebook_size_x = over_sampling_x * Mx
    codebook_size_y = over_sampling_y * My
    codebook_size_z = over_sampling_z * Mz

    # X-direction codebook
    theta_qx = np.linspace(0, np.pi - 1e-6, codebook_size_x)
    F_CBx = np.zeros((Mx, codebook_size_x), dtype=complex)
    for i in range(len(theta_qx)):
        F_CBx[:, i] = np.sqrt(1/Mx) * np.exp(-1j * kd * antx_index * np.cos(theta_qx[i]))

    # Y-direction codebook
    theta_qy = np.linspace(0, np.pi - 1e-6, codebook_size_y)
    F_CBy = np.zeros((My, codebook_size_y), dtype=complex)
    for i in range(len(theta_qy)):
        F_CBy[:, i] = np.sqrt(1/My) * np.exp(-1j * kd * anty_index * np.cos(theta_qy[i]))

    # Z-direction codebook
    theta_qz = np.linspace(0, np.pi - 1e-6, codebook_size_z)
    F_CBz = np.zeros((Mz, codebook_size_z), dtype=complex)
    for i in range(len(theta_qz)):
        F_CBz[:, i] = np.sqrt(1/Mz) * np.exp(-1j * kd * antz_index * np.cos(theta_qz[i]))

    # Combine codebooks
    F_CBxy = np.kron(F_CBy, F_CBx)
    F_CB = np.kron(F_CBz, F_CBxy)

    # Beam indices
    beams_x = np.arange(1, codebook_size_x + 1)
    beams_y = np.arange(1, codebook_size_y + 1)
    beams_z = np.arange(1, codebook_size_z + 1)

    Mxx_Ind = np.tile(beams_x, codebook_size_y * codebook_size_z)
    Myy_Ind = np.tile(np.tile(beams_y, codebook_size_x).reshape(-1, order='F'), codebook_size_z)
    Mzz_Ind = np.tile(beams_z, codebook_size_x * codebook_size_y)

    Tx = np.stack([Mxx_Ind, Myy_Ind, Mzz_Ind], axis=1)
    all_beams = Tx

    return F_CB, all_beams

def plot_pattern(vec, fig=None, ax=None, label=None):
    """
    Plot the radiation pattern of a measurement vector

    Parameters:
    vec: Input measurement vector (column vector)
    fig: Optional figure to use (if None, creates new figure)
    ax: Optional axis to use (if None, creates new axis)
    label: Optional label for the plot
    """

    # Ensure vec is a column vector
    if vec.ndim == 1:
        vec = vec.reshape(-1, 1)

    My = vec.shape[0]
    over_sampling_y = 1000

    # Generate codebook
    F, _ = UPA_codebook_generator(1, My, 1, 1, over_sampling_y, 1, 0.5)

    # Angular sampling
    theta_s = np.linspace(0, np.pi - 1e-6, over_sampling_y * My)

    # Project vector onto codebook
    projection = F.conj().T @ vec  # F.H @ vec (Hermitian transpose)
    proj = np.abs(projection)**2

    # Find maximum
    argidx = np.argmax(proj)

    # Create polar plot if not provided
    if fig is None and ax is None:
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, polar=True)

        # Configure plot (only for new plots)
        ax.grid(True, alpha=0.25)
        ax.set_rlabel_position(90)

    # Plot each beam
    for n in range(vec.shape[1]):
        if label is not None and vec.shape[1] == 1:
            ax.plot(theta_s, proj[:, n], linewidth=2, label=label)
        else:
            ax.plot(theta_s, proj[:, n], linewidth=2, label=f'Beam {n+1}' if label is None else label)

    # Add legend if multiple beams or if we have labels
    if vec.shape[1] > 1 or label is not None:
        ax.legend()

    return fig, ax

if __name__ == "__main__":
    # Example usage
    # Generate a simple test vector
    test_vec = np.random.randn(8, 1) + 1j * np.random.randn(8, 1)
    fig, ax = plot_pattern(test_vec)
    plt.show()
