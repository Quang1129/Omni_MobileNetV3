import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .data_transform import *

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465],
						std=[0.2023, 0.1994, 0.2010]):
	single_img = False
	if tensor.ndimension() == 3:
		single_img = True
		tensor = tensor[None, :, :, :]

	if not tensor.ndimension() == 4:
	    raise TypeError('Tensor should be 4D')
    
	mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
	ret = tensor.mul(std).add(mean)
 
	return ret[0] if single_img else ret

def imshow(img, title):
    img = denormalize(img)
    npimg = img.numpy()
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title) 
    plt.axis('off')
    plt.show()

from torchvision.utils import make_grid

class TinyImageNet(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.device = args.device
        self.num_workers = args.num_workers
        self.train_path = args.train_dir
        self.test_path = args.val_dir
        
        self.load()
    

    def _transforms(self):
        train_transform = albumentations_transforms(p = 1.0, is_train = True)
        test_transform = albumentations_transforms(p = 1.0, is_train = False)
        return train_transform, test_transform

    def _dataset(self):
        train_transform, test_transform = self._transforms()
        
        train_set = ImageFolder(root = self.train_path, 
                                transform = train_transform)
        test_set = ImageFolder(root = self.test_path,
                               transform = test_transform)
        return train_set, test_set

    def load(self):
        train_set, test_set = self._dataset()
        self.num_classes = len(train_set.class_to_idx)
        self.data_labels = train_set.class_to_idx
        
        # Dataloader Arguments
        dataloader_args = dict(
            shuffle = True,
            batch_size = self.batch_size
        )

        if self.device == 'cuda':
            dataloader_args.update(
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                pin_memory = True
            )

        self.train_loader = DataLoader(train_set, **dataloader_args)
        self.test_loader = DataLoader(test_set, **dataloader_args)
       

    def show_samples(self):
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)


        index = []
        num_img = min(self.num_classes, 10)

        for i in range(num_img):
            for j in range(len(labels)):
                if labels[j] == i:
                    index.append(j)
                    break

        if len(index) < num_img:
            for j in range(len(labels)):
                if len(index) == num_img:
                    break
                if j not in index:
                    index.append(j)

        print(index)
        print(num_img)

        imshow(torchvision.utils.make_grid(images[index],
				nrow=5, scale_each=True), "Sample train data")
