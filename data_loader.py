import numpy as np
import torch
import torch.utils.data as tdata
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class Cifar10(object):
    @classmethod
    def get_train_valid_loader(cls,
                               data_dir,
                               batch_size,
                               augment,
                               random_seed,
                               subset_size=1,
                               valid_size=0.1,
                               shuffle=True,
                               num_workers=4,
                               pin_memory=False):
        error_msg1 = "[!] valid_size should be in the range [0, 1]."
        error_msg2 = "[!] subset_size should be in the range [0, 1]."
        assert ((valid_size >= 0) and (valid_size <= 1)), error_msg1
        assert ((subset_size >= 0) and (valid_size <= 1)), error_msg2

        normalize = transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768],
        )

        # define transforms
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(16),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        # load the dataset
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

        num_train = len(train_dataset)
        split_subset = int(np.floor(subset_size * num_train))
        indices_subset = list(range(split_subset))
        split_valid = int(np.floor(valid_size * split_subset))

        if shuffle:
            np.random.shuffle(indices_subset)

        train_idx, valid_idx = indices_subset[split_valid:], indices_subset[:split_valid]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = tdata.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = tdata.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return train_loader, valid_loader

    @classmethod
    def get_train_loader(cls,
                         data_dir,
                         batch_size,
                         augment,
                         random_seed,
                         shuffle=True,
                         num_workers=4,
                         pin_memory=False):
        normalize = transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768],
        )

        # define transforms
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(16),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        # load the dataset
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))

        if shuffle:
            np.random.shuffle(indices)

        train_idx = indices
        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = tdata.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return train_loader

    @classmethod
    def get_test_loader(cls,
                        data_dir,
                        batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=False):
        normalize = transforms.Normalize(
            mean=[0.49139968, 0.48215827, 0.44653124],
            std=[0.24703233, 0.24348505, 0.26158768],
        )

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        data_loader = tdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return data_loader


class Cifar100(object):
    @classmethod
    def get_train_valid_loader(cls,
                               data_dir,
                               batch_size,
                               augment,
                               random_seed,
                               subset_size=1,
                               valid_size=0.1,
                               shuffle=True,
                               num_workers=4,
                               pin_memory=False):
        error_msg1 = "[!] valid_size should be in the range [0, 1]."
        error_msg2 = "[!] subset_size should be in the range [0, 1]."
        assert ((valid_size >= 0) and (valid_size <= 1)), error_msg1
        assert ((subset_size >= 0) and (valid_size <= 1)), error_msg2

        normalize = transforms.Normalize(
            mean=[0.50707516, 0.48654887, 0.44091784],
            std=[0.26733429, 0.25643846, 0.27615047],
        )

        # define transforms
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(16),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        # load the dataset
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        valid_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=valid_transform,
        )

        num_train = len(train_dataset)
        split_subset = int(np.floor(subset_size * num_train))
        indices_subset = list(range(split_subset))
        split_valid = int(np.floor(valid_size * split_subset))

        if shuffle:
            np.random.shuffle(indices_subset)

        train_idx, valid_idx = indices_subset[split_valid:], indices_subset[:split_valid]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = tdata.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = tdata.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return train_loader, valid_loader

    @classmethod
    def get_train_loader(cls,
                         data_dir,
                         batch_size,
                         augment,
                         random_seed,
                         shuffle=True,
                         num_workers=4,
                         pin_memory=False):
        normalize = transforms.Normalize(
            mean=[0.50707516, 0.48654887, 0.44091784],
            std=[0.26733429, 0.25643846, 0.27615047],
        )

        # define transforms
        if augment:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                Cutout(16),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        # load the dataset
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))

        if shuffle:
            np.random.shuffle(indices)

        train_idx = indices
        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = tdata.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return train_loader

    @classmethod
    def get_test_loader(cls,
                        data_dir,
                        batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=False):
        normalize = transforms.Normalize(
            mean=[0.50707516, 0.48654887, 0.44091784],
            std=[0.26733429, 0.25643846, 0.27615047],
        )

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        dataset = datasets.CIFAR100(
            root=data_dir, train=False,
            download=True, transform=transform,
        )

        data_loader = tdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return data_loader
