from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

#dsname = "mnist"
class IndexedDataset(Dataset):
    """
    this class can return the index of the example in the dataset
    for example-forgetting indexing purpose (saving forgetting statistics)
    """
    def __init__(self,dsname):
        self.ds = None
        if dsname == "mnist":
            self.ds = datasets.MNIST(root='data/mnist',
                                        download=True,
                                        train=True,
                                        transform=transforms.Compose([
                                        transforms.Pad(padding=2, fill=0, padding_mode='constant'),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
]))
        elif dsname == "cifar10":
            self.ds = datasets.CIFAR10(root='data/cifar10',
                                        download=True,
                                        train=True,
                                        transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
]))
        else :
            raise Exception('dsname must be "mnist" or "cifar10", dsname was: {}'.format(dsname))

    def __getitem__(self, index):
        data, target = self.ds[index]
        return data, target, index

    def __len__(self):
        return len(self.ds)


