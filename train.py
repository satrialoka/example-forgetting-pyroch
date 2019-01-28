import torch
import torchvision
import dataloaders
import model
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

train_err = []
test_err = []
train_acc = []
test_acc = []

def train(data_train_loader,criterion,optimizer,epoch,recordacc):
        #net.train()
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
                        accuracy += calcacc(output,labels)
        train_loss = train_loss/len(data_train_loader)
        if recordacc == True : 
                accuracy = accuracy/len(data_train_loader)
                return train_loss, accuracy
        return train_loss

def test(data_test_loader,criterion,optimizer,epoch,recordacc):
        #net.eval()
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
                for i, (images, labels, indexes) in enumerate(data_test_loader):
                        output = net(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()
                        if recordacc == True :
                                accuracy += calcacc(output,labels)
                test_loss = test_loss/len(data_test_loader)
                if recordacc == True : 
                        accuracy = accuracy/len(data_test_loader)
                        return test_loss, accuracy
                return test_loss

def calcacc(output,labels):
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        acc = torch.mean(equals.type(torch.FloatTensor))
        return acc

def train_test():
        epoch = 100
        for e in range(epoch):
                train_loss, train_acuracy = train(train_loader,criterion,optimizer,e+1,True)
                test_loss, test_acuracy = test(test_loader,criterion,optimizer,e+1,True)
                print('Epoch %d, Training Loss/Acc: %f//%f, Testing Loss/Acc: %f//%f' % (e+1,train_loss,train_acuracy,test_loss,test_acuracy))

                train_err.append(train_loss)
                test_err.append(test_loss)
                train_acc.append(train_acuracy)
                test_acc.append(test_acuracy)

if __name__ == '__main__':        
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))
        train_dataset = dataloaders.IndexedDataset("mnist",istrain=True)
        train_loader = DataLoader(train_dataset,
                    batch_size=256,
                    shuffle=True,
                    num_workers=0)
        test_dataset = dataloaders.IndexedDataset("mnist",istrain=False)
        test_loader = DataLoader(test_dataset,
                    batch_size=256,
                    shuffle=False,
                    num_workers=0)
        net = model.LeNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)

        #train(loader,criterion,optimizer,30,True)
        train_test()

        
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
