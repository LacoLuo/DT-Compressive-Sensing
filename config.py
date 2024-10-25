class configurations(object):
    def __init__(self):
        # Dataset param.
        self.real_data_root = "DeepMIMO/Datasets/Boston5G_3p5_target"
        self.synth_data_root = "DeepMIMO/Datasets/Boston5G_3p5_nofoliage_shifted_1"
        self.train_csv = "train_data_idx.csv"
        self.test_csv = "test_data_idx.csv"
        self.N_BS = 32
        self.N_MS = 1
        self.M_BS = 8
        self.M_MS = 1

        # Train param.
        self.num_train_data = 10240
        self.batch_size = 32
        self.learning_rate = 1e-2
        self.num_epochs = 200
        self.gpu = 0