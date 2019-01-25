import torch
import torchvision
import dataloaders
import model
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader





def train(data_train_loader,model,criterion,optimizer,epoch):
        net.train()
        loss_list, batch_list = [], []
        for e in range(epoch):
                for i, (images, labels, indexes) in enumerate(data_train_loader):
                        optimizer.zero_grad()
                        output = net(images)
                        loss = criterion(output, labels)

                        loss_list.append(loss.detach().cpu().item())
                        batch_list.append(i+1)

                        if i % 10 == 0:
                                print('Train - Epoch %d, Batch: %d, Loss: %f' % (e, i, loss.detach().cpu().item()))

                        loss.backward()
                        optimizer.step()


if __name__ == '__main__':
        
        dataset = dataloaders.IndexedDataset("mnist")
        loader = DataLoader(dataset,
                    batch_size=32,
                    shuffle=True,
                    num_workers=0)


        net = model.LeNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=2e-3)

        train(loader,net,criterion,optimizer,5)

        
"""
        data,label = dataset.ds[1]
        print(data)
        for batch_idx, (data, target, idx) in enumerate(loader):
                print('Batch idx {}, dataset index {}'.format(batch_idx, idx))
"""
