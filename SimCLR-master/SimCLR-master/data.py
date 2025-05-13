from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFilter
from PIL import Image


class GaussianSmoothing(object):
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(self.radius))


def cifar_train_transforms():
    all_transforms = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


def cifar_test_transforms():
    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


def mnist_train_transforms():
    # Defining the augmentations
    all_transforms = transforms.Compose([
        transforms.RandomAffine(degrees=15,
                                translate=[0.1, 0.1],
                                scale=[0.9, 1.1],
                                shear=15),
        transforms.ToTensor()
    ])
    return all_transforms


def mnist_test_transforms():
    all_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    return all_transforms


class CIFAR10C(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFAR10C, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            xi = self.transform(img)
            xj = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return xi, xj, target


class CustomDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Args:
            root_dirs (list): List of directories with images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.image_paths = []

        # Collect all image paths
        for root_dir in root_dirs:
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                        self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            xi = self.transform(image)
            xj = self.transform(image)

        return xi, xj



class MNISTC(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(MNISTC, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            xi = self.transform(img)
            xj = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return xi, xj, target


class Loader(object):
    def __init__(self, dataset_ident, file_path, download, batch_size, train_transform, test_transform, target_transform, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        loader_map = {
            'CIFAR10C': CIFAR10C,
            'CIFAR10': datasets.CIFAR10,
            'MNIST': datasets.MNIST,
            'MNISTC': MNISTC
        }

        num_class = {
            'CIFAR10C': 10,
            'CIFAR10': 10,
            'MNIST': 10,
            'MNISTC': 10
        }

        # Get the datasets
        train_dataset, test_dataset = self.get_dataset(loader_map[dataset_ident], file_path, download,
                                                       train_transform, test_transform, target_transform)
        # Set the loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        tmp_batch = self.train_loader.__iter__().__next__()[0]
        self.img_shape = list(tmp_batch.size())[1:]
        self.num_class = num_class[dataset_ident]

    @staticmethod
    def get_dataset(dataset, file_path, download, train_transform, test_transform, target_transform):

        # Training and Validation datasets
        train_dataset = dataset(file_path, train=True, download=download,
                                transform=train_transform,
                                target_transform=target_transform)

        test_dataset = dataset(file_path, train=False, download=download,
                               transform=test_transform,
                               target_transform=target_transform)

        return train_dataset, test_dataset
