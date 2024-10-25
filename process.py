import os 
import sys 
import numpy as np 
from tqdm import tqdm
from collections import OrderedDict

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from data_feed import DataFeed
from model import ConvPrecoder_MISO
from utils import save_model, load_model

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def build_model(config, device):
    dims = [config.N_BS, config.N_MS, config.M_BS, config.M_MS]
    num_classes = config.N_BS
    
    # Build the model
    model = ConvPrecoder_MISO(dims, num_classes)
    summary(model, input_size=(config.batch_size, 1, config.N_BS*config.N_MS), dtypes=[torch.cfloat])
    
    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.load_model_path:
        model = load_model(model, config.load_model_path)
    model = model.to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100], gamma=0.1
    )
    
    return model, optimizer, scheduler

def train_process(config, seed=0):
    setup_seed(seed)
    device = torch.device(f'cuda:{config.gpu}')
    
    # Get output directory ready
    if not os.path.isdir(config.store_model_path):
        os.makedirs(config.store_model_path)
    
    # Create a summary writer with the specified folder name
    writer = SummaryWriter(os.path.join(config.store_model_path, 'summary'))
    
    # Prepare training data
    if config.DT and config.finetune:
        finetune_real_dataset = DataFeed(config.real_data_root, config.train_csv, num_data_point=config.num_finetune_data)
        finetune_synth_dataset = DataFeed(config.synth_data_root, config.train_csv, num_data_point=config.num_train_data)
        finetune_dataset = ConcatDataset([finetune_real_dataset, finetune_synth_dataset])
        train_loader = DataLoader(finetune_dataset, batch_size=config.batch_size, shuffle=True)
    elif config.DT:
        train_loader = DataLoader(
            DataFeed(config.synth_data_root, config.train_csv, num_data_point=config.num_train_data),
            batch_size=config.batch_size,
            shuffle=True
        )
    else:
        train_loader = DataLoader(
            DataFeed(config.real_data_root, config.train_csv, num_data_point=config.num_train_data),
            batch_size=config.batch_size,
            shuffle=True
        )
    
    val_feed = DataFeed(config.real_data_root, config.test_csv)
    val_loader = DataLoader(val_feed, batch_size=1024)
    
    # Build model
    model, optimizer, scheduler = build_model(config, device)
    print("Finish building model")
    
    # Define loss function
    loss_function = nn.CrossEntropyLoss()
    
    # Training
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        with tqdm(train_loader, unit="batch", file=sys.stdout) as tepoch:
            for i, data in enumerate(tepoch, 0):
                tepoch.set_description(f"Epoch {epoch}")
                
                # Get the inputs
                channel, beam_idx = data[0].to(device), data[1].to(device)
            
                # Set the parameter gradients to zero
                optimizer.zero_grad()
                
                # Forward + Backward + Optimize
                pred_idx = model(channel)
                loss = loss_function(pred_idx, beam_idx)
                # Orthonormal regularization
                if config.M_BS > 1: 
                    kernel = torch.squeeze(model.BS_filters.weight).to(device)
                    loss_ort = torch.norm(torch.matmul(kernel, torch.t(torch.conj(kernel))) - torch.eye(config.M_BS).to(device), 'fro')
                    loss += 1e-1 * loss_ort
                
                loss.backward()
                optimizer.step()
                
                acc = torch.sum(pred_idx.argmax(1) == beam_idx).item() / config.batch_size
                
                # Normalize the conv. weights
                with torch.no_grad():
                    # model.BS_filters.weight /= torch.norm(model.BS_filters.weight, dim=2, keepdim=True) # Unit norm
                    model.BS_filters.weight /= torch.abs(model.BS_filters.weight) # constant-modulus
                    model.BS_filters.weight *= 1/(config.N_BS**.5) # Unit norm
                
                # Pring statistics
                running_loss = (loss.item() + i * running_loss) / (i + 1)
                running_acc = (acc + i * running_acc) / (i + 1)
                log = OrderedDict()
                log["loss"] = "{:.6e}".format(running_loss)
                log["acc"] = running_acc
                tepoch.set_postfix(log)
            scheduler.step()

        # Validation
        val_loss = 0
        val_acc = 0
        if epoch >= 190:
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    # Get the inputs
                    channel, beam_idx = data[0].to(device), data[1].to(device)
                    
                    # Forward
                    pred_idx = model(channel)
                    loss = loss_function(pred_idx, beam_idx)               
                    acc = torch.sum(pred_idx.argmax(1) == beam_idx).item() / beam_idx.shape[0]
                    
                    val_loss += loss * beam_idx.shape[0]
                    val_acc += acc * beam_idx.shape[0]
                
                val_loss /= len(val_feed)
                val_acc /= len(val_feed)
            print("val_loss={:.6e}".format(val_loss), flush=True)
            print("val_acc={:.6f}".format(val_acc), flush=True)
        
        # Write summary
        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Loss/test", val_loss, epoch)
        writer.add_scalar("Acc/train", running_acc, epoch)
        writer.add_scalar("Acc/test", val_acc, epoch)
                
    writer.close()    
    
    # Save model   
    if config.store_model_path:
        save_model(model, config.store_model_path)   
        
    return val_acc 
    
def test_process(config):
    device = torch.device(f'cuda:{config.gpu}')
    
    # Prepare test data
    test_feed = DataFeed(config.real_data_root, config.test_csv)
    test_loader = DataLoader(test_feed, batch_size=1024)
    
    # Build model
    model, _, _ = build_model(config, device)
    print("Finish building model")
    
    # Define loss function
    loss_function = nn.CrossEntropyLoss(reduction='none')
    
    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = []
        test_data_idx = []
        
        for data in test_loader:
            # Get the inputs
            channel, beam_idx, data_idx = data[0].to(device), data[1].to(device), data[2].to(device)
            
            # Forward
            pred_idx = model(channel)
            loss = loss_function(pred_idx, beam_idx) 
            
            test_loss.append(loss.numpy(force=True))
            test_data_idx.append(data_idx.numpy(force=True))
            
        test_loss = np.concatenate(test_loss)
        test_data_idx = np.concatenate(test_data_idx)
    
    # Save measurment vectors
    BS_meas_vectors = torch.squeeze(model.BS_filters.weight)
    if BS_meas_vectors.dim() == 2:
        BS_meas_vectors = torch.transpose(BS_meas_vectors, 0, 1)
    else:
        BS_meas_vectors = torch.unsqueeze(BS_meas_vectors, 1)

    return {
        "BS_meas_vecs": BS_meas_vectors.numpy(force=True),
        "test_loss_all": test_loss,
        "test_data_idx": test_data_idx
    }