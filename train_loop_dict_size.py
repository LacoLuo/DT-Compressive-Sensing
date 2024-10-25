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
    config.synth_data_root = "DeepMIMO/Datasets/Boston5G_3p5_nofoliage_shifted_1" # Change scenario
    
    dict_sizes = [1, 2, 4, 8, 16, 32]
    num_training = 10
    
    np.random.seed(10)
    seeds = np.random.randint(0, 10000, size=(num_training,))
    
    all_avg_acc = []
    for dict_size in dict_sizes:
        config.M_BS = dict_size
        
        if config.DT:
            scenario = os.path.basename(config.synth_data_root)
            store_model_root_dir = os.path.join(
                "results",
                scenario,
                f"results_DT_{config.N_BS}x1_dict_size_{dict_size}",
            )
        else:
            scenario = os.path.basename(config.real_data_root)
            store_model_root_dir = os.path.join(
                "results",
                scenario,
                f"results_real_{config.N_BS}x1_dict_size_{dict_size}",
            )
        
        all_acc = []
        for i in range(0, num_training):
            sub_dir = f"train_{i+1}"
            config.store_model_path = os.path.join(store_model_root_dir, sub_dir)
            
            val_acc = train_process(config, seed=seeds[i])
            all_acc.append(val_acc)
        all_avg_acc.append(np.asarray(all_acc))
    
    all_avg_acc = np.stack(all_avg_acc, 0)
    
    if config.DT:
        savemat(
            os.path.join("results", scenario, f"all_avg_acc_train_on_DT_{config.N_BS}x1_dict_size.mat"),
            {"all_avg_acc_train_on_DT": all_avg_acc},
        )
    else:
        savemat(
            os.path.join("results", scenario, f"all_avg_acc_train_on_real_{config.N_BS}x1_dict_size.mat"),
            {"all_avg_acc_train_on_real": all_avg_acc},
        )
        
    print("done")