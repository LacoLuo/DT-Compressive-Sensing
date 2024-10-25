import torch 
import datetime

def save_model(model, store_model_path):
  now = datetime.datetime.now().strftime("%H_%M_%S")
  date = datetime.date.today().strftime("%y_%m_%d")
  comment = "_".join([now, date])
  
  torch.save(model.state_dict(), f'{store_model_path}/{comment}.ckpt')
  return

def load_model(model, load_model_path):
  print(f'Load model from {load_model_path}')
  model.load_state_dict(torch.load(f'{load_model_path}'))
  return model

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
  return torch.unsqueeze(uu, 1)
