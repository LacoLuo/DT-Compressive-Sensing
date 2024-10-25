import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class ConvPrecoder_MISO(nn.Module):
    # Input: (batch_size, 1, N_BS*N_MS) vectorized channel
    # Output: (batch_size, N_BS) codeword
    def __init__(self, dims, num_classes):
        super().__init__()
        
        N_BS = dims[0]
        N_MS = dims[1]
        M_BS = dims[2]
        M_MS = dims[3]
        
        self.BS_filters = nn.Conv1d(1, M_BS, kernel_size=N_BS, stride=N_BS, 
                                    padding='valid', dtype=torch.cfloat, bias=True)
        
        self.linear_layers = [nn.Linear(2*M_BS*M_MS, N_BS*N_MS),
                              nn.Linear(N_BS*N_MS, N_BS*N_MS)]
        self.linear_layers = nn.ModuleList(self.linear_layers)
        
        self.pred_layer = nn.Linear(N_BS*N_MS, num_classes)
        
    def forward(self, x):
        out = self.BS_filters(x) # (batch_size, M_BS, 1)
        
        out = torch.view_as_real(out) # (batch_size, M_BS, 1, 2)
        out = torch.reshape(out, (out.shape[0], -1)) # (batch_size, 2*M_BS)
        
        for i in range(len(self.linear_layers)):
            out = self.linear_layers[i](out)
            out = torch.sigmoid(out)
            
        out = self.pred_layer(out)
        
        return out 

if __name__ == "__main__":
    device = torch.device("cuda:0")
    dims = [32, 1, 8, 1]
    num_classes = 32
    
    model = ConvPrecoder_MISO(dims, num_classes)
    summary(model, input_size=(16, 1, 32), dtypes=[torch.cfloat])
    print(model.BS_filters.weight.size())

    x = model.BS_filters.weight
    x = 1/(32**.5) * x / torch.abs(x)
    print(torch.norm(x[0, 0, :]))
    
    
    # Test normalizing
    x = torch.randn(2, 1, 2, dtype=torch.cfloat).to(device)
    x = x / torch.norm(x, dim=2, keepdim=True)
    print(torch.norm(x[0, 0, :]))
    
    
    """
    # Test Gram-Schmidt
    def gram_schmidt(vv):
        def projection(u, v):
            return (v * torch.conj(u)).sum() / (u * torch.conj(u)).sum() * u

        nk = vv.size(0)
        uu = torch.zeros_like(vv, device=vv.device)
        uu[0, :] = vv[0, :].clone()
        for k in range(1, nk):
            vk = vv[k, :].clone()
            uk = 0
            for j in range(0, k):
                uj = uu[j, :].clone()
                uk = uk + projection(uj, vk)
            uu[k, :] = vk - uk
        for k in range(nk):
            uk = uu[k, :].clone()
            uu[k, :] = uk / uk.norm()
        return uu
    tmp = gram_schmidt(torch.squeeze(model.BS_filters.weight))
    print(torch.abs(torch.matmul(tmp, torch.t(torch.conj(tmp)))))
    """
    
    x = torch.randn(2, 1, 32, dtype=torch.cfloat).to(device)
    y = model.forward(x)
    print(y.size())
    
    
    