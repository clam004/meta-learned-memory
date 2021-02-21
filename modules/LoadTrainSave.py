import torch

def save_model(model,name):
    torch.save(model.state_dict(),name)
    
def load_model(model,name):
    model.load_state_dict(torch.load(name))