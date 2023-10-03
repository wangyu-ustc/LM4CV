import os
import pdb
import torch
import pandas as pd
from torchvision.datasets.folder import default_loader
import numpy as np
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
from PIL import Image


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://s3.us-west-2.amazonaws.com/caltechdata/96/97/8384-3670-482e-a3dd-97ac171e8a10/data?response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3B%20filename%3DCUB_200_2011.tgz&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARCVIVNNAP7NNDVEA%2F20221218%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221218T091222Z&X-Amz-Expires=60&X-Amz-SignedHeaders=host&X-Amz-Signature=5b57318c941bf095e654e5f650df3e3c4ce68defd540d3ce9841a30bfad7acde'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, mode='train', num_classes=200, transform=None, loader=default_loader, download=True,
                 with_attributes=False, attributes_version='v0'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        # self.train = train
        self.num_classes = num_classes
        self.mode = mode
        self.with_attributes = with_attributes

        if with_attributes:
            assert os.path.exists(root + f'/CUB_200_2011/attributes_{attributes_version}.txt'), print(
                f"No attributes found, please run description.py for attributes_{attributes_version} first!")
            with open(root + f'/CUB_200_2011/attributes_{attributes_version}.txt') as file:
                self.attributes = file.read().strip().split("\n")
            file.close()

            # Add clip prediction as the prior
            with open(root + f'/CUB_200_2011/clip_classification.txt') as file:
                clip_classification = [eval(l) for l in file.read().split("\n")]
            file.close()

            with open('./data/CUB_200_2011/classes.txt', 'r') as file:
                classes = [cla.split(".")[1].replace("_", ' ') for cla in file.read().strip().split("\n")]
            file.close()

            self.attributes = [f"{classes[pred - 1]} with {attr}" for attr, pred in
                               zip(self.attributes, clip_classification)]

        self._load_metadata()
        # if download:
        #     self._download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.mode == 'train':
            if self.with_attributes:
                self.attributes = np.array(self.attributes)[self.data.is_training_img == 1]
            self.data = self.data[self.data.is_training_img == 1]

        elif self.mode == 'test':
            if self.with_attributes:
                self.attributes = np.array(self.attributes)[self.data.is_training_img == 0]
            self.data = self.data[self.data.is_training_img == 0]

        if self.with_attributes:
            self.attributes = np.array(self.attributes)[self.data.target <= self.num_classes]
        self.data = self.data[self.data.target <= self.num_classes]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.with_attributes:
            return img, target, self.attributes[idx]
        else:
            return img, target


class ImagenetA(Dataset):
    def __init__(self, root, preprocess=None):
        self.preprocess = preprocess
        meta = torch.load("data/Imagenet/meta.bin")
        cls2foldername = {cls:folder for folder,cls in meta[0].items()}

        classnames = open("data/Imagenet/classes.txt", "r").read().split("\n")
        self.files = []
        self.labels = []

        for idx, cla in enumerate(classnames):
            if tuple(cla.split(", ")) in cls2foldername:
                if os.path.exists(os.path.join(root, cls2foldername[tuple(cla.split(", "))])):
                    for file in os.listdir(os.path.join(root, cls2foldername[tuple(cla.split(", "))])):
                        self.files.append(os.path.join(root, cls2foldername[tuple(cla.split(", "))], file))
                        self.labels.append(idx)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = self.preprocess(img) if self.preprocess is not None else img
        return img, self.labels[idx]





class WaterBirds(Dataset):
    """
    CUB dataset (already cropped and centered).
    Note: metadata_df is one-indexed.
    """

    def __init__(self, root_dir,
                 target_name='waterbird',
                 confounder_names='forest2water',
                 augment_data=False,
                 model_type=None,
                 transform=None,
                 eval_transform=None):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data

        # self.data_dir = os.path.join(
        #     self.root_dir,
        #     'data',
        #     '_'.join([self.target_name] + self.confounder_names))
        self.data_dir = self.root_dir

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1
        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype('int')

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        # filter shared groups in train
        train_indices = np.where(self.split_array == 0)[0]
        train_remove_indices = np.where((self.group_array[train_indices] == 1) | (self.group_array[train_indices] == 2))
        self.split_array[train_indices[train_remove_indices]] = 3

        # filter shared groups in test
        test_indices = np.where(self.split_array == 2)[0]
        test_remove_indices = np.where((self.group_array[test_indices] == 0) | (self.group_array[test_indices] == 3))
        self.split_array[test_indices[test_remove_indices]] = 4

        # Set transform
        # if model_attributes[self.model_type]['feature_type']=='precomputed':
        #     self.features_mat = torch.from_numpy(np.load(
        #         os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
        #     self.train_transform = None
        #     self.eval_transform = None
        # else:
        self.features_mat = None
        if transform is not None:
            self.train_transform = transform
        else:
            self.train_transform = get_transform_cub(
                self.model_type,
                train=True,
                augment_data=augment_data)

        if eval_transform is not None:
            self.eval_transform = eval_transform
        elif transform is not None:
            self.eval_transform = transform
        else:
            self.eval_transform = get_transform_cub(
                self.model_type,
                train=False,
                augment_data=augment_data)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        # if model_attributes[self.model_type]['feature_type']=='precomputed':
        if False:
            x = self.features_mat[idx, :]
        else:
            img_filename = os.path.join(
                self.data_dir,
                self.filename_array[idx])
            img = Image.open(img_filename).convert('RGB')
            # Figure out split and transform accordingly
            if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
                img = self.train_transform(img)
            elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
                  self.eval_transform):
                img = self.eval_transform(img)
            # Flatten if needed
            # if model_attributes[self.model_type]['flatten']:
            # assert img.dim()==3
            # img = img.view(-1)
            x = img

        return x, y, g

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train', 'val', 'test'), split + ' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets


def get_transform_cub(model_type, train, augment_data):
    scale = 256.0 / 224.0
    # target_resolution = model_attributes[model_type]['target_resolution']
    # target_resolution = (224, 224)
    target_resolution = (336, 336)
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


class CatDog(Dataset):
    base_folder = 'PetImages/images'

    def __init__(self, root, mode='train', num_classes=200, transform=None, loader=default_loader, download=True,
                 with_attributes=False, attributes_version='v0'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        # self.train = train
        self.num_classes = num_classes
        self.mode = mode
        self.with_attributes = with_attributes

        if with_attributes:
            assert os.path.exists(root + f'/PetImages/attributes_{attributes_version}.txt'), print(
                f"No attributes found, please run description.py for attributes_{attributes_version} first!")
            with open(root + f'/PetImages/attributes_{attributes_version}.txt') as file:
                self.attributes = file.read().strip().split("\n")
            file.close()

            # Add clip prediction as the prior
            # with open(root + f'/PetImages/clip_classification.txt') as file:
            #     clip_classification = [eval(l) for l in file.read().split("\n")]
            # file.close()

            # with open('./data/PetImages/classes.txt', 'r') as file:
            #     classes = [cla.split(" ")[1] for cla in file.read().strip().split("\n")]
            # file.close()

            # self.attributes = [f"{classes[pred - 1]} with {attr}" for attr, pred in zip(self.attributes, clip_classification)]

        self._load_metadata()
        # if download:
        #     self._download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'PetImages', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'PetImages', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'PetImages', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.mode == 'train':
            self.attributes = np.array(self.attributes)[self.data.is_training_img == 1]
            self.data = self.data[self.data.is_training_img == 1]

        elif self.mode == 'test':
            self.attributes = np.array(self.attributes)[self.data.is_training_img == 0]
            self.data = self.data[self.data.is_training_img == 0]

        if self.with_attributes:
            self.attributes = np.array(self.attributes)[self.data.target <= self.num_classes]
        self.data = self.data[self.data.target <= self.num_classes]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        if self.with_attributes:
            return img, target, self.attributes[idx]
        else:
            return img, target


class ScoreDataset(Dataset):
    def __init__(self, scores, targets):
        self.scores = torch.tensor(scores)
        self.targets = torch.tensor(targets)

    def __getitem__(self, idx):
        return self.scores[idx], self.targets[idx]

    def __len__(self):
        return len(self.scores)


class OnlineScoreDataset(Dataset):
    def __init__(self, attribute_embeddings, features, targets):
        self.features = torch.tensor(features).float()
        self.targets = torch.tensor(targets).float()
        self.attribute_embeddings = attribute_embeddings

    def __getitem__(self, idx):
        feature = self.features[idx]
        scores = feature.unsqueeze(0) @ self.attribute_embeddings.float().T
        scores = scores.squeeze()
        return scores, self.targets[idx]
        # return self.scores[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets, group_array=None):
        self.features = torch.tensor(features)
        self.targets = torch.tensor(targets)
        self.group_array = group_array

    def __getitem__(self, idx):
        if self.group_array is not None:
            return self.features[idx], self.targets[idx], self.group_array[idx]
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)