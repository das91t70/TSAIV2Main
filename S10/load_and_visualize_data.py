import matplotlib.pyplot as plt
class CIFAR10DataOps:

  def get_train_transforms(self):
      return albumentations.Compose([
    albumentations.PadIfNeeded(min_height=36, min_width=36, p=1),
    albumentations.RandomCrop(32,32),
    albumentations.HorizontalFlip(p=0.5), # flip left right - horizontal flip
    albumentations.CoarseDropout(max_holes=1, max_height = 8, max_width=8, min_holes = 1, min_height=8, min_width=8, fill_value=(0.49139968, 0.48215827 ,0.44653124,), mask_fill_value = None, p=0.2),
    albumentations.augmentations.transforms.Normalize((0.49139968, 0.48215827 ,0.44653124,), (0.24703233,0.24348505,0.26158768,)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])
  def get_test_transforms(self):
      return albumentations.Compose([
        albumentations.augmentations.transforms.Normalize((0.49139968, 0.48215827 ,0.44653124,), (0.24703233,0.24348505,0.26158768,)),
        albumentations.pytorch.transforms.ToTensorV2()
        ])
  def __init__(self, train):
    if train:
      self.dataset = datasets.CIFAR10('./data', train=train, download=True, transform=Tranforms(self.get_train_transforms()))
    else:
      self.dataset = datasets.CIFAR10('./data', train=train, download=True, transform=Tranforms(self.get_test_transforms()))

  
  def load_dataset(self, dataloader_args):
    data_loader = torch.utils.data.DataLoader(self.dataset, **dataloader_args)
    return data_loader

  def visualize_data_in_batch(self, data_loader, start, end):
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    self.imshow(make_grid(images[start:end+1]), [classes[index] for index in labels.tolist()[start:end+1]])



  def imshow(self, img, labels):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(labels)
    plt.show()

