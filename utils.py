import h5py
import torch
import shutil

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File('./saved_models'+fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, './saved_models/'+task_id+filename)
    if is_best:
        shutil.copyfile('./saved_models/'+task_id+filename, './saved_models/'+task_id+'model_best.pth.tar')            

def tv_loss(y):
    #loss = torch.sum(torch.abs(y[:-1, :, :, :] - y[1:, :, :, :])) + torch.sum(torch.abs(y[:, :-1, :, :] - y[:, 1:, :, :]))
    loss = torch.sum((y[:-1, :, :, :] - y[1:, :, :, :])**2) + torch.sum((y[:, :-1, :, :] - y[:, 1:, :, :])**2)
    return loss

