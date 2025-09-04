import os 
import argparse
from scipy.io import savemat
from process import test_process
from config import configurations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Inference")
    parser.add_argument(
            "-l", "--load_model_path", required=True, type=str,
            help="path of pretrained model")
    args = parser.parse_args()

    config = configurations()
    config.load_model_path = args.load_model_path
    print('config:\n', vars(config))

    config.M_BS = 4
    
    ret = test_process(config)
    BS_meas_vectors = ret["BS_meas_vecs"]
    mdic = {"BS_meas_vecs": BS_meas_vectors}
    savemat(f"./meas_vecs_M_{config.M_BS}.mat", mdic)