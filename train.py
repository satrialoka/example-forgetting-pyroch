import torch
import torchvision
import dataloaders
import model
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def train(data_train_loader,model,criterion,optimizer,epoch,recordacc):
        net.train()
        train_loss = 0
        accuracy = 0
        for i, (images, labels, indexes) in enumerate(data_train_loader):
                optimizer.zero_grad()
                output = net(images)
                loss = criterion(output, labels)
                

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if recordacc == True :
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        acc = torch.mean(equals.type(torch.FloatTensor))
                        accuracy += acc

                if i % 200 == 0:
                        print('Train - Epoch %d, Batch: %d, Loss: %f, Acc: %f' % (epoch, i, loss.detach().cpu().item(),acc))
        train_loss = train_loss/len(data_train_loader)
        if recordacc == True : 
                accuracy = accuracy/len(data_train_loader)
                return train_loss, accuracy
        return train_loss

def test(data_test_loader,model,criterion,optimizer,epoch,recordacc):
        net.test()
        test_loss = 0
        accuracy = 0
        for i, (images, labels, indexes) in enumerate(data_test_loader):
                optimizer.zero_grad()
                output = net(images)
                loss = criterion(output, labels)
                test_loss += loss.item()

                if recordacc == True :
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        acc = torch.mean(equals.type(torch.FloatTensor))
                        accuracy += acc

                if i % 200 == 0:
                        print('Test - Epoch %d, Batch: %d, Loss: %f, Acc: %f' % (epoch, i, loss.detach().cpu().item(),acc))
        test_loss = test_loss/len(data_test_loader)
        if recordacc == True : 
                accuracy = accuracy/len(data_test_loader)
                return test_loss, accuracy
        return test_loss





if __name__ == '__main__':        
        dataset = dataloaders.IndexedDataset("mnist")
        loader = DataLoader(dataset,
                    batch_size=256,
                    shuffle=True,
                    num_workers=0)
        net = model.LeNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=2e-3)

        train(loader,net,criterion,optimizer,30,True)

        
"""
        from torchsummary import summary
        import torch
        import torch.nn as nn
        

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        model = net.to(device)

        summary(model, input_size=(1, 28, 28))


        data,label = dataset.ds[1]
        print(data)
        for batch_idx, (data, target, idx) in enumerate(loader):
                print('Batch idx {}, dataset index {}'.format(batch_idx, idx))
"""
