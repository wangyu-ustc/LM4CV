'''
evaluate zero-shot performance
'''
import os
import sys
import pdb
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn

import pandas as pd
import torchvision
import transformers
from collections import defaultdict
from dataset import ScoreDataset, ImagenetA
import sklearn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader


import clip

def clean_label(true_labels):
    true_labels = np.array(true_labels)
    if np.min(true_labels) > 0:
        true_labels -= np.min(true_labels)
    return true_labels

def get_labels(dataset):
    if dataset == 'cub':
        with open("./data/CUB_200_2011/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)
        train_test_split = pd.read_csv(os.path.join('./data/', 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        train_test_split = train_test_split['is_training_img'].values
        train_indices = np.where(train_test_split == 1)
        test_indices = np.where(train_test_split == 0)
        train_labels, test_labels = true_labels[train_indices], true_labels[test_indices]

    elif dataset == 'cifar100':
        with open("./data/cifar-100-python/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-10000], true_labels[-10000:]

    elif dataset == 'cifar10' or dataset == 'cifar10-p':
        with open("./data/cifar-10-batches-py/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-10000], true_labels[-10000:]

    elif dataset == 'food':
        with open("./data/food-101/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)
        train_labels, test_labels = true_labels[:-25250], true_labels[-25250:]

    elif dataset == 'flower':
        with open("./data/flowers-102/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        train_labels, test_labels = true_labels[:-1020], true_labels[-1020:]

    elif dataset == 'cars':

        with open("./data/stanford_cars/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 8144 + 8041
        train_labels, test_labels = true_labels[:8144], true_labels[-8041:]


    elif dataset == 'imagenet':
        with open("./data/Imagenet/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

    elif dataset == 'imagenet-a':
        with open("./data/Imagenet/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        def filter_labels(labels):
            idxes = np.where((labels < 398) & (labels!=69))
            return labels[idxes]

        train_labels = filter_labels(train_labels)

        train_labels[np.where(train_labels>69)] -= 1

        testset = ImagenetA(root='./data/imagenet-a')
        test_labels = testset.labels


    elif dataset == 'imagenet-animal':
        with open("./data/Imagenet/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 50000 + 1281167

        train_labels, test_labels = true_labels[:1281167], true_labels[-50000:]

        train_labels, test_labels = np.array(train_labels), np.array(test_labels)

        def filter_labels(labels):
            idxes = np.where((labels < 398) & (labels!=69))
            return labels[idxes]

        train_labels = filter_labels(train_labels)
        test_labels = filter_labels(test_labels)

        train_labels[np.where(train_labels>69)] -= 1
        test_labels[np.where(test_labels>69)] -= 1

    elif dataset == 'oxford_pets':
        with open("./data/oxford-iiit-pet/image_class_labels.txt", 'r') as file:
            true_labels = [eval(line.split(" ")[1]) for line in file.read().strip().split("\n")]
        file.close()
        true_labels = clean_label(true_labels)

        assert len(true_labels) == 3680 + 3669

        train_labels, test_labels = true_labels[:3680], true_labels[-3669:]

    else:
        raise NotImplementedError


    return train_labels, test_labels


def get_image_dataloader(dataset, preprocess, preprocess_eval=None, shuffle=False):

    if dataset == 'cub':
        # Load dataset
        from dataset import Cub2011
        train_dataset = Cub2011(root='./data/', mode='train', transform=preprocess)
        test_dataset = Cub2011(root='./data/', mode='test', transform=preprocess)

        print("Train dataset:", len(train_dataset))
        print("Test dataset:", len(test_dataset))

        train_loader = DataLoader(train_dataset, batch_size=96, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=96, shuffle=False)

    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=preprocess)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    elif dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    elif dataset == 'food':
        trainset = torchvision.datasets.Food101(root='./data/', split='train', download=True, transform=preprocess)
        testset = torchvision.datasets.Food101(root='./data/', split='test', download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    elif dataset == 'flower':
        trainset = torchvision.datasets.Flowers102(root='./data/', split='train', download=True, transform=preprocess)
        testset = torchvision.datasets.Flowers102(root='./data/', split='val', download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    elif dataset == 'cars':
        trainset = torchvision.datasets.StanfordCars(root='./data/', split='train', download=True, transform=preprocess)
        testset = torchvision.datasets.StanfordCars(root='./data/', split='test', download=True, transform=preprocess)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    elif dataset == 'imagenet' or dataset == 'imagenet-animal' or dataset == 'imagenet-a':
        trainset = torchvision.datasets.ImageNet(root='./data/Imagenet', split='train', transform=preprocess)
        testset = torchvision.datasets.ImageNet(root='./data/Imagenet', split='val', transform=preprocess)

        if dataset == 'imagenet-animal' or dataset == 'imagenet-a':

            def filter_dataset(dataset):
                targets = np.array(dataset.targets)
                idxes = np.where((targets < 398) & (targets!=69))
                dataset.targets = targets[idxes].tolist()
                dataset.samples = [dataset.samples[i] for i in idxes[0]]

            filter_dataset(trainset)
            filter_dataset(testset)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=False)

        if dataset == 'imagenet-a':
            testset = ImagenetA(root='./data/imagenet-a', preprocess=preprocess)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

    elif dataset == 'oxford_pets':
        trainset = torchvision.datasets.OxfordIIITPet(root='./data/', split='trainval', transform=preprocess, download=True)
        testset = torchvision.datasets.OxfordIIITPet(root='./data/', split='test', transform=preprocess, download=True)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)


    else:
        raise NotImplementedError


    return train_loader, test_loader

def get_output_dim(dataset):
    return len(np.unique(get_labels(dataset)[0]))


def get_folder_name(dataset):
    if dataset == 'cub':
        return 'cub'
    elif dataset == 'cifar100':
        return 'cifar-100-python'
    elif dataset == 'cifar10':
        return 'cifar-10-batches-py'
    elif dataset == 'flower':
        return "flowers-102"
    elif dataset == 'food':
        return "food-101"
    elif dataset == 'imagenet' or dataset == 'imagenet-animal':
        return "Imagenet"
    elif dataset == 'imagenet-a':
        return "imagenet-a"
    elif dataset == 'cars':
        return 'stanford_cars'
    elif dataset == 'oxford_pets':
        return "oxford-iiit-pet"
    elif dataset == "waterbirds":
        return 'waterbird_complete95_forest2water2'
    else:
        raise NotImplementedError


def get_attributes(cfg):

    if cfg['attributes'] == 'random':
        '''
        Generate random attributes
        '''
        import urllib.request
        import random

        word_url = "https://www.mit.edu/~ecprice/wordlist.10000"
        response = urllib.request.urlopen(word_url)
        long_txt = response.read().decode()
        word_list = long_txt.splitlines()

        print(len(word_list))

        random_words = []
        for i in range(512):
            words = random.choices(word_list, k=random.randint(1, 5))
            random_words.append(' '.join(words))

        attributes = random_words
        return attributes

    elif cfg['attributes'] == 'cub':
        return open("./data/CUB_200_2011/cub_attributes.txt", 'r').read().strip().split("\n")

    elif cfg['attributes'] == 'flower':
        return open("./data/flowers-102/flower_attributes.txt", 'r').read().strip().split("\n")

    elif cfg['attributes'] == 'food':
        return open("./data/food-101/food_attributes.txt", 'r').read().strip().split("\n")
    
    elif cfg['attributes'] == 'cars':
        return open("./data/stanford_cars/cars_attributes.txt", 'r').read().strip().split("\n")

    elif cfg['attributes'] == 'imagenet':
        return open("./data/Imagenet/imagenet_attributes.txt", 'r').read().strip().split("\n")

    elif cfg['attributes'] == 'imagenet-animal':
        return open("./data/Imagenet/imagenet_animal_attributes.txt", 'r').read().strip().split("\n")

    elif cfg['attributes'] == 'cifar10':
        return open("./data/cifar-10-batches-py/cifar10_attributes.txt", 'r').read().strip().split("\n")

    elif cfg['attributes'] == 'cifar100':
        return open("./data/cifar-100-python/cifar100_attributes.txt", 'r').read().strip().split("\n")

    elif cfg['attributes'] == 'oxford_pets':
        return open("./data/oxford-iiit-pet/oxford_pets_attributes.txt", 'r').read().strip().split("\n")

    else:
        raise NotImplementedError



def get_prefix(cfg):
    if cfg['attributes'] == 'cbm':
        return ""
    if cfg['dataset'] == 'cub' or cfg['dataset'] == 'waterbirds':
        # return "A photo of a bird with "
        return "The bird has "
    elif cfg['dataset'] == 'cifar100':
        return "A photo of an object with "
    elif cfg['dataset'] == 'cifar10':
        # return "A photo of an object with "
        return "A blur photo of an object with"
    elif cfg['dataset'] == 'cifar10-p':
        return "A photo of an object with "
    elif cfg['dataset'] == 'flower':
        return "A photo of the flower with "
    elif cfg['dataset'] == 'food':
        return "A photo of the food with "
    elif cfg['dataset'] == 'cars':
        return "A photo of the car with "
    elif cfg['dataset'] == 'oxford_pets':
        return "A photo of the animal with "
    elif cfg['dataset'] == 'imagenet':
        return "A photo of an object with "
    elif cfg['dataset'] in ['imagenet-animal', 'imagenet-a']:
        return "A photo of an animal with "
    else:
        raise NotImplementedError
