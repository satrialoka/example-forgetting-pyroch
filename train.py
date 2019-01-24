#%%
import torch
import torchvision
import dataloaders
import model
from torchsummary import summary


dataset = dataloaders.IndexedDataset("mnist")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.LeNet()
model = model.to(device)

summary(model, input_size=(1, 32, 32))

#%%



