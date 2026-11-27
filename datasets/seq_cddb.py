import json
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Tuple

from datasets.utils import set_default_from_args
from utils import smart_joint
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders, getAllLoaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
from utils.prompt_templates import templates


import numpy as np
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

def get_raw_data_and_targets(dataset_path, image_size=32, grayscale=False):
    """
    Chuyển đổi một ImageFolder dataset thành 2 numpy array: data và targets.
    
    Parameters:
    - dataset_path (str): Đường dẫn tới thư mục dataset.
    - image_size (int): Kích thước ảnh đầu ra (width và height).
    - grayscale (bool): Nếu True, trả về ảnh xám (N, H, W). Nếu False, trả về ảnh màu (N, H, W, C).
    
    Returns:
    - data (numpy.ndarray): Array chứa các ảnh chưa transform, dtype=float32.
    - targets (numpy.ndarray): Array chứa các nhãn tương ứng, dtype=int64.
    """
    # Khởi tạo ImageFolder mà không truyền transform vào (để lấy ảnh raw gốc)
    dataset = ImageFolder(root=dataset_path)
    
    raw_images = []
    targets_list = []
    
    # Lặp qua từng đường dẫn ảnh và nhãn (label) của chúng
    # dataset.samples chứa list các tuple: (đường dẫn_ảnh, class_index)
    for img_path, target in tqdm(dataset.samples, desc="Converting data from torchvision.ImageFolder to Numpy array..."):
        # Đọc ảnh gốc bằng PIL
        with Image.open(img_path) as img:
            # Chuyển đổi định dạng ảnh
            if grayscale:
                img = img.convert("L")  # Chuyển sang ảnh xám (L)
            else:
                img = img.convert("RGB") # Chuyển sang ảnh màu nếu cần
                
            # Resize ảnh về kích thước mong muốn (giống cách CIFAR10 có size cố định)
            img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
            
            # Chuyển thành numpy array
            img_array = np.array(img, dtype=np.float32)
            
            raw_images.append(img_array)
            targets_list.append(target)
            
    # Gộp danh sách ảnh thành 1 numpy array duy nhất
    data = np.stack(raw_images, axis=0)
    
    # Chuyển targets thành numpy array với dtype=int64
    targets = np.array(targets_list, dtype=np.int64)
    
    return data, targets

class CDDB(Dataset):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.not_aug_transform = transforms.ToTensor()
        data, targets = get_raw_data_and_targets(root, image_size=224, grayscale=False)
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray((img * 255).astype(np.uint8).transpose(1, 2, 0), mode='RGB')

        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialCDDBHard(ContinualDataset):
    NAME = ""
    SETTING = "class-il"
    N_TASKS = 0
    SIZE = (224, 224)
    MEAN, STD = 0, 0
    TRAIN_PATH = ""
    TEST_PATH = ""
    
    N_CLASSES = 2
    N_CLASSES_PER_TASK = 2 
    LABELS = ["0_real", "1_fake"]
    
    TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(size=SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std=0),
        ])
        return transform
    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=0, std=0)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(0, 0)
        return transform
    
    def get_data_loaders(self):
        train_dataset = CDDB(self.TRAIN_PATH,train=True,
                               download=True, transform=self.TRANSFORM)
        test_dataset = CDDB(self.TEST_PATH, train=False, download=True,
                              transform=self.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)

        return train, test

    def get_all_data_loaders(self):
        train_dataset = CDDB(self.TRAIN_PATH,train=True,
                               download=True, transform=self.TRANSFORM)
        test_dataset = CDDB(self.TEST_PATH, train=False, download=True,
                              transform=self.TEST_TRANSFORM)

        train, test = getAllLoaders(train_dataset, test_dataset, self)
        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = [x.replace('_', ' ') for x in self.LABELS]  # .split('___')[-1]
        classes = fix_class_names_order(classes, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_prompt_templates():
        return templates['cifar100']


    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 5

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128


    






