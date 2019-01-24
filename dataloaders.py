from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class IndexedDataset(Dataset):
    """
    this class can return the index of the example in the dataset
    for example-forgetting indexing purpose (saving forgetting statistics)
    """
    def __init__(self,dsname="mnist"):
        self.dataset = None
        if dsname == "mnist":
            self.dataset = datasets.MNIST(root='data/mnist',
                                          download=True,
                                          train=True,
                                          transform=transforms.ToTensor)
        elif dsname == "cifar10":
            self.dataset = datasets.CIFAR10(root='data/cifar10',
                                          download=True,
                                          train=True,
                                          transform=transforms.ToTensor)
        else :
            raise Exception('dsname must be "mnist" or "cifar10", dsname was: {}'.format(dsname))

    def __getitem__(self,index):

        data,target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


