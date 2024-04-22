import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import albumentations
import albumentations.pytorch
import numpy as np
import matplotlib.pyplot as plt
class CIFAR10DataOps:

  # register CIFAR10 dataset as either train or test data
  def __init__(self, train):
    if train:
      self.dataset = datasets.CIFAR10('./data', train=train, download=True, transform=Transforms(self.get_train_transforms()))
    else:
      self.dataset = datasets.CIFAR10('./data', train=train, download=True, transform=Transforms(self.get_test_transforms()))

  # get transformations used for training dataset
  def get_train_transforms(self):
      return albumentations.Compose([
    albumentations.PadIfNeeded(min_height=36, min_width=36, p=1),
    albumentations.RandomCrop(32,32),
    albumentations.HorizontalFlip(p=0.5), # flip left right - horizontal flip
    albumentations.CoarseDropout(max_holes=1, max_height = 8, max_width=8, min_holes = 1, min_height=8, min_width=8, fill_value=(0.49139968, 0.48215827 ,0.44653124,), mask_fill_value = None, p=0.2),
    albumentations.augmentations.transforms.Normalize((0.49139968, 0.48215827 ,0.44653124,), (0.24703233,0.24348505,0.26158768,)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

  # get transformations used for testing dataset
  def get_test_transforms(self):
      return albumentations.Compose([
        albumentations.augmentations.transforms.Normalize((0.49139968, 0.48215827 ,0.44653124,), (0.24703233,0.24348505,0.26158768,)),
        albumentations.pytorch.transforms.ToTensorV2()
        ])

  # load train/test dataset with dataloader arguments
  def load_dataset(self, dataloader_args):
    data_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_args)
    return data_loader
    
  # visualize dataloader in a batch
  def visualize_data_in_batch(self, data_loader, start, end):
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    self.imshow(make_grid(images[start:end+1]), [classes[index] for index in labels.tolist()[start:end+1]])


  # shpw images
  def imshow(self, img, labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(labels)
    plt.show()

# transforms class - used for albumentations
class Transforms:
    def __init__(self, transforms: albumentations.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))
