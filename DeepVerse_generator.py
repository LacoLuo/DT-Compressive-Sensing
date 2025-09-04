import os
import numpy as np
from pprint import pprint
from scipy.io import savemat
from deepverse import ParameterManager, Dataset 

def UPA_codebook_generator_DFT(Mx, My, Mz, over_sampling_x=1, over_sampling_y=1, over_sampling_z=1, ant_spacing=0.5):
    """
    Generates a DFT-based codebook for a Uniform Planar Array (UPA).

    Args:
        Mx (int): Number of antenna elements on x-axis.
        My (int): Number of antenna elements on y-axis.
        Mz (int): Number of antenna elements on z-axis.
        over_sampling_x (int, optional): Oversampling factor along x-axis. Defaults to 1.
        over_sampling_y (int, optional): Oversampling factor along y-axis. Defaults to 1.
        over_sampling_z (int, optional): Oversampling factor along z-axis. Defaults to 1.
        ant_spacing (float, optional): Antenna spacing. Defaults to 0.5 (half wavelength).

    Returns:
        tuple: A tuple containing:
            - F_CB (np.ndarray): The generated UPA codebook.
            - all_beams (np.ndarray): A matrix of all beam indices for each axis.
    """
    
    antx_index = np.arange(Mx)
    anty_index = np.arange(My)
    antz_index = np.arange(Mz)
    
    codebook_size_x = over_sampling_x * Mx
    codebook_size_y = over_sampling_y * My
    codebook_size_z = over_sampling_z * Mz
    
    theta_qx = np.arange(codebook_size_x) * (2 * np.pi / codebook_size_x)
    F_CBx = (1 / np.sqrt(Mx)) * np.exp(-1j * np.outer(antx_index, theta_qx))
    
    theta_qy = np.arange(codebook_size_y) * (2 * np.pi / codebook_size_y)
    F_CBy = (1 / np.sqrt(My)) * np.exp(-1j * np.outer(anty_index, theta_qy))
    
    theta_qz = np.arange(codebook_size_z) * (2 * np.pi / codebook_size_z)
    F_CBz = (1 / np.sqrt(Mz)) * np.exp(-1j * np.outer(antz_index, theta_qz))
    
    F_CBxy = np.kron(F_CBy, F_CBx)
    F_CB = np.kron(F_CBz, F_CBxy)
    
    beams_x = np.arange(1, codebook_size_x + 1)
    beams_y = np.arange(1, codebook_size_y + 1)
    beams_z = np.arange(1, codebook_size_z + 1)
    
    Mxx_Ind = np.tile(beams_x, codebook_size_y * codebook_size_z)
    Myy_Ind = np.tile(np.tile(beams_y, codebook_size_x).reshape(-1, order='F'), codebook_size_z)
    Mzz_Ind = np.tile(beams_z, codebook_size_x * codebook_size_y)
    
    all_beams = np.column_stack((Mxx_Ind, Myy_Ind, Mzz_Ind))
    
    return F_CB, all_beams

if __name__ == "__main__":
    # Path to the configuration file 
    scenario_name = 'DT1'
    config_path = f"scenarios/{scenario_name}/param/config.m"

    # Initialize ParameterManager and load parameters
    param_manager = ParameterManager(config_path)
    param_manager.params["scenes"] = list(range(2411))
    # param_manager.params["scenes"] = list(range(10))
    param_manager.params["radar"]["enable"] = False
    param_manager.params["comm"]["enable"] = True
    param_manager.params["camera"] = False
    param_manager.params["lidar"] = False
    param_manager.params["position"] = True

    param_manager.params["comm"]["OFDM"]["bandwidth"] = 30e3/1e9
    param_manager.params["comm"]["OFDM"]["subcarriers"] = 1
    param_manager.params["comm"]["OFDM"]["selected_subcarriers"] = list(range(0, 1))

    param_manager.params["comm"]["bs_antenna"]["shape"] = [32, 1]
    param_manager.params["comm"]["bs_antenna"]["rotation"] = [0, 0, 0]
    param_manager.params["comm"]["bs_antenna"]["FoV"] = [180, 180]

    param_manager.params["comm"]["ue_antenna"]["shape"] = [1, 1]
    param_manager.params["comm"]["ue_antenna"]["rotation"] = [0, 0, 0]
    param_manager.params["comm"]["ue_antenna"]["FoV"] = [360, 180]

    # pprint(param_manager.params)

    # Generate a dataset
    dataset = Dataset(param_manager)

    # Generate DFT codebook
    num_tx_ant = param_manager.params["comm"]["bs_antenna"]["shape"][0]
    num_rx_ant = param_manager.params["comm"]["ue_antenna"]["shape"][0]
    F_CB, _ = UPA_codebook_generator_DFT(num_tx_ant, 1, 1)
    print("Codebook Shape:", F_CB.shape)

    # Define the output variables
    num_scene = len(dataset.params['scenes'])
    all_beam_idx = np.zeros((num_scene, 1))
    all_channel = np.zeros((num_scene, num_rx_ant, num_tx_ant), dtype=np.complex64)
    
    for i in range(num_scene):
        channel = dataset.get_sample('comm-ue', index=i, bs_idx=0, ue_idx=0).coeffs # [rx, tx, subcarriers]
        channel = np.squeeze(channel, axis=2) # [rx, tx]
        gain = abs(channel @ np.conj(F_CB)) # [rx, codebook_size]
        beam_idx = np.argmax(gain, axis=1)
        all_beam_idx[i] = beam_idx 
        all_channel[i, :, :] = channel

    print("All Beam Indices Shape:", all_beam_idx.shape)
    print("All Channel Shape:", all_channel.shape)

    # Define the output directory
    output_dir = f"datasets/{scenario_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the output variables
    savemat(
        os.path.join(output_dir, "dataset.mat"), 
        {"all_beam_idx": all_beam_idx, "all_channel": all_channel})