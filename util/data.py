import numpy as np
import argparse
import os
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda
from torch.utils.data.sampler import WeightedRandomSampler,RandomSampler

def get_data(args: argparse.Namespace): 
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """
    if args.dataset =='CUB-200-2011':
        # 3090
        # 8 classes
        # return get_birds(True, r'data/CUB_200_2011\dataset1\train-6400', r'data/CUB_200_2011\dataset1\train-6400', r'data/CUB_200_2011\dataset1\test-800')

        # return get_birds(True, r'data/CUB_200_2011\dataset1\test-800',r'data/CUB_200_2011\dataset1\test-800', r'data/CUB_200_2011\dataset1\test-800')
        # 7 classes
        # return get_birds(True, r'data/dataset\train-6400', r'data/dataset\train-6400', r'data/dataset\test-800')
        # return get_birds(True, r'data/CUB_200_2011\dataset1\train-6400', r'data/CUB_200_2011\dataset1\train-6400', r'data/CUB_200_2011\dataset1\test-800')
        # 3060 8class
        return get_birds(True, r'data/dataset_8class\train-6400',
                         r'data/CUB_200_2011/dataset1\train-6400',
                         # r'data/CUB_200_2011/dataset1\test-800'
                         r'data/test'
                         # r'data/CUB_200_2011/dataset\test-800'
                         )

    if args.dataset == 'CARS':
        return get_cars(True, r'data/CARS/dataset/test', r'data/CARS/dataset/test', r'data/CARS/dataset/test')
    raise Exception(f'Could not load data set "{args.dataset}"!')

def get_dataloaders(args: argparse.Namespace):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, shape = get_data(args)
    c, w, h = shape
    print('trainset:',len(trainset))
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=cuda
                                              )
    print('trainset:', len(trainloader))
    projectloader = torch.utils.data.DataLoader(projectset,
                                                #    batch_size=args.batch_size,
                                                batch_size=int(args.batch_size / 4),
                                                # make batch size smaller to prevent out of memory errors during projection
                                                shuffle=False,
                                                pin_memory=cuda
                                                )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, projectloader, testloader, classes, c

def get_dataloaders1(args: argparse.Namespace,w_init):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, classes, shape = get_data(args)

    c, w, h = shape
    print('trainset:',len(trainset))
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    sampler = RandomSampler(trainset,num_samples=40960,replacement=True)
    print('w_init_data:',w_init)
# #     # Bagging trees
# #     # trainloader = torch.utils.data.DataLoader(trainset,
# #     #                                           batch_size=args.batch_size,
# #     #                                           shuffle=True,
# #     #                                           pin_memory=cuda,
# #     #                                           sampler=sampler
# #     #                                           )
# #     # Single tree
# #     # trainloader = torch.utils.data.DataLoader(trainset,
# #     #                                           batch_size=args.batch_size,
# #     #                                           shuffle=True,
# #     #                                           pin_memory=cuda,
# #     #
# #     #                                           )
# #     # Ensemble trees
    sampler = WeightedRandomSampler(weights=w_init, num_samples=51200,replacement=False)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=cuda,
                                              sampler=sampler
                                              )
    sampler1 = WeightedRandomSampler(weights=w_init, num_samples=51200, replacement=True)
    trainloader1 = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=cuda,
                                              sampler=sampler1
                                              )
    print('trainloader:',len(trainloader))
    projectloader = torch.utils.data.DataLoader(projectset,
                                            #    batch_size=args.batch_size,
                                              batch_size=int(args.batch_size/4), #make batch size smaller to prevent out of memory errors during projection
                                              shuffle=False,
                                              pin_memory=cuda
                                              )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda
                                             )
    print("Num classes (k) = ", len(classes), flush=True)
    # print(len(trainloader))
    return trainloader, trainloader1,projectloader, testloader, classes, c
    # return trainloader, projectloader, testloader, classes, c

def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])
    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),    # Perspective transform (simulates 3D view changes)
            # transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
            transforms.RandomHorizontalFlip(), # Horizontal flip
            transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),# Affine transform (rotation + translation + shear)
            ]),
            transforms.ToTensor(),    # 3. Convert to tensor
            normalize,   # 4. Normalize
        ])
    else:
        transform = transform_no_augment
    # from torch.utils.data import Dataset
    # from torchvision.datasets import ImageFolder
    # import os
    # from PIL import Image
    #
    # class CustomLabelDataset(Dataset):
    #     def __init__(self, root_dir, transform=None, label_mapping=None):
    #         """
    #         Custom label dataset
    #         :param root_dir: dataset root directory
    #         :param transform: data preprocessing
    #         :param label_mapping: label mapping dict {class name: new label}
    #         """
    #         # Load the base dataset with ImageFolder
    #         self.base_dataset = ImageFolder(root_dir, transform=None)
    #         self.transform = transform
    #
    #         # Create a mapping from folder names to new labels
    #         self.label_mapping = label_mapping or {}
    #
    #         # Verify that all classes have a mapping
    #         for class_name in self.base_dataset.classes:
    #             if class_name not in self.label_mapping:
    #                 raise ValueError(f"No label mapping defined for class '{class_name}'")
    #
    #     def __len__(self):
    #         return len(self.base_dataset)
    #
    #     def __getitem__(self, idx):
    #         # Get the original image and label
    #         img_path, original_label = self.base_dataset.samples[idx]
    #         img = Image.open(img_path).convert('RGB')
    #
    #         # Get the class name
    #         class_name = self.base_dataset.classes[original_label]
    #
    #         # Apply the label mapping
    #         new_label = self.label_mapping[class_name]
    #
    #         # Apply the transform
    #         if self.transform:
    #             img = self.transform(img)
    #
    #         return img, new_label
    #
    #     @property
    #     def classes(self):
    #         """Return the list of class names"""
    #         return self.base_dataset.classes

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform)
    # Define the desired label mapping
    # label_mapping = {
    #     "002.DEB": 1,  # map folder class0 to label 1
    #     "005.MUS": 4,  # map folder class1 to label 4
    #     "006.NORM": 5,  # map folder class2 to label 5
    #     "008.TUM": 7  # map folder class3 to label 7
    # }
    # label_mapping = {
    #     "DEB": 1,  # map folder class0 to label 1
    #     "MUS": 4,  # map folder class1 to label 4
    #     "NORM": 5,  # map folder class2 to label 5
    #     "TUM": 7  # map folder class3 to label 7
    # }
    # testset = CustomLabelDataset(
    #     test_dir,
    #     transform=transform_no_augment,  # preprocessing pipeline
    #     label_mapping=label_mapping
    # )

    for i in range(5):
        tensor, label = testset[i]
        print(f"Image {i + 1} processed size: {tensor.shape} (CxHxW)")
        # Usually torch.Size([3, 224, 224]) for ResNet
    classes = trainset.classes
    for i in range(len(classes)):
        classes[i]=classes[i].split('.')[1]
    return trainset, projectset, testset, classes, shape


def get_cars(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
            transforms.RandomOrder([
            transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
            transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=15,shear=(-2,2)),
            ]),
            transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transform_no_augment

    trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
    testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    classes = trainset.classes
    
    return trainset, projectset, testset, classes, shape


