import os 
import numpy as np
from scipy.io import savemat

from process import train_process
from config import configurations

if __name__ == '__main__':
    config = configurations()
    config.load_model_path = None
    config.finetune = None
    config.DT = True # Choose target or DT training data
    config.synth_data_root = "datasets/DT1" # Change scenario  
    
    if config.DT:
        scenario = os.path.basename(config.synth_data_root)
        store_model_path = os.path.join(
            "results",
            scenario,
            f"results_DT_{config.N_BS}x1_dict_size_{config.M_BS}",
        )
    else:
        scenario = os.path.basename(config.real_data_root)
        store_model_path = os.path.join(
            "results",
            scenario,
            f"results_real_{config.N_BS}x1_dict_size_{config.M_BS}",
        )
    config.store_model_path = store_model_path
    val_acc = train_process(config, seed=0)
           
    print("done")