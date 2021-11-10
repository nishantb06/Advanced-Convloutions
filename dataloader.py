import torch
import torchvision
import torchvision.transforms as transforms


torch.manual_seed(2)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class args():
    def __init__(self,device = 'cpu' ,use_cude = False) -> None:
        self.batch_size = 128
        self.device = device
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
                                        
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,
                                          shuffle=True, **args().kwargs)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=args().batch_size,
                                         shuffle=False, **args().kwargs)