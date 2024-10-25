import os 
import numpy as np
from scipy.io import savemat

from process import train_process
from config import configurations

if __name__ == '__main__':
    config = configurations()
    config.DT = True
    config.finetune = True
    config.M_BS = 8
    config.num_epochs = 200
    config.learning_rate = 1e-3
    num_training = 10
    
    config.synth_data_root = "DeepMIMO/Datasets/Boston5G_3p5_nofoliage_shifted_1" # Change scenario
    scenario = os.path.basename(config.synth_data_root)
    folder_root = os.path.join("results", scenario)
    ckpt_folder_root = os.path.join(folder_root, f"results_DT_32x1_dict_size_{config.M_BS}")

    np.random.seed(0)
    seeds = np.random.randint(0, 10000, size=(num_training,))

    all_avg_acc = []
    for num_finetune_data in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]:
        config.num_finetune_data = num_finetune_data
        store_model_root_dir = os.path.join(
            folder_root,
            f"results_finetune_DT_{config.N_BS}x1_dict_size_{config.M_BS}_num_data_{num_finetune_data}")

        all_acc = []
        for i in range(0, num_training):
            sub_dir = f"train_{i+1}"
            config.store_model_path = os.path.join(store_model_root_dir, sub_dir)
            
            ckpt_folder_path = os.path.join(ckpt_folder_root, sub_dir)
            model_name = [
                filename for filename in os.listdir(ckpt_folder_path)
                if filename.endswith(".ckpt")
            ]
            config.load_model_path = os.path.join(ckpt_folder_path, model_name[0])
            
            val_acc = train_process(config, seed=seeds[i])
            all_acc.append(val_acc)
        all_avg_acc.append(np.asarray(all_acc))
    
    all_avg_acc = np.stack(all_avg_acc, 0)
    
    savemat(
        os.path.join(folder_root, f"all_avg_acc_train_on_DT_{config.N_BS}x1_finetune.mat"),
        {"all_avg_acc_train_on_DT": all_avg_acc},
    )

    print("done")