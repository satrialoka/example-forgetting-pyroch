import torch
import torchvision
import dataloaders
import model
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

train_err = []
test_err = []
train_acc = []
test_acc = []
batchsize = 64

train_dataset = dataloaders.IndexedDataset("fmnist",istrain=True)
train_loader = DataLoader(train_dataset,
                          batch_size=batchsize,
                          shuffle=True,
                          num_workers=0)
test_dataset = dataloaders.IndexedDataset("fmnist",istrain=False)
test_loader = DataLoader(test_dataset,
                         batch_size=batchsize,
                         shuffle=False,
                         num_workers=0)


def train(data_train_loader,criterion,optimizer,epoch,recordacc):
        net.train()
        istrain = True
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
                        accuracy += calcacc(output,labels,indexes,istrain,epoch)
        train_loss = train_loss/len(data_train_loader)
        if recordacc == True : 
                accuracy = accuracy/len(data_train_loader)
                return train_loss, accuracy
        return train_loss

def test(data_test_loader,criterion,optimizer,epoch,recordacc):
        net.eval()
        istrain = False
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
                for i, (images, labels, indexes) in enumerate(data_test_loader):
                        output = net(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()
                        if recordacc == True :
                                accuracy += calcacc(output,labels,indexes,istrain,epoch)
                test_loss = test_loss/len(data_test_loader)
                if recordacc == True : 
                        accuracy = accuracy/len(data_test_loader)
                        return test_loss, accuracy
                return test_loss

prev_acci = np.zeros(len(train_loader)*batchsize)
T = np.zeros(len(train_loader)*batchsize)
learnt = np.zeros(len(train_loader)*batchsize)

def calcacc(output,labels,indexes, istrain,epoch):
        global prev_acci
        global T
        global learnt        

        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        if istrain :
            for i in range (len(indexes)):

                acci = equals[i]
                if prev_acci[indexes[i]] > acci :
                    T[indexes[i]] = T[indexes[i]]+1
                if prev_acci[indexes[i]] < acci :
                    learnt[indexes[i]] = 1                    
                    
                prev_acci[indexes[i]] = acci
        acc = torch.mean(equals.type(torch.FloatTensor))
        return acc

        

def train_test():
    
    
        epoch = 500
        for e in range(epoch):
                train_loss, train_acuracy = train(train_loader,criterion,optimizer,e+1,True)
                test_loss, test_acuracy = test(test_loader,criterion,optimizer,e+1,True)
                print('Epoch %d, Training Loss/Acc: %f//%f, Testing Loss/Acc: %f//%f' % (e+1,train_loss,train_acuracy,test_loss,test_acuracy))

                train_err.append(train_loss)
                test_err.append(test_loss)
                train_acc.append(train_acuracy)
                test_acc.append(test_acuracy)

        plt.plot(train_err)
        plt.plot(test_err)
        plt.savefig('error.png')
        
        plt.close()
        
        plt.plot(train_acc)
        plt.plot(test_acc)
        plt.savefig('acc.png')
        
        np.save("results\forgetting_stat.npy",T)
        np.save("results\learnt.npy",learnt)
if __name__ == '__main__':        
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(0))       
        net = model.Fenet()
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(net.parameters(), lr=0.01)
        optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.5)

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
