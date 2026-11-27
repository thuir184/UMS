from datasets.seq_cddb import SequentialCDDBHard
import torchvision.transforms as transforms
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode

class SequentialCDDB_wild(SequentialCDDBHard):
    NAME = "seq-cddb-wild"
    SETTING = "class-il"
    N_TASKS = 5
    MEAN, STD = [0.4318, 0.3439, 0.32], [0.2418, 0.2133, 0.2183]
    TRAIN_PATH = "data/cddb-hard/wild/train"
    TEST_PATH = "data/cddb-hard/wild/val"
    SIZE = (224, 224)
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
        MEAN, STD = [0.4318, 0.3439, 0.32], [0.2418, 0.2133, 0.2183]
        transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        return transform
    @staticmethod
    def get_normalization_transform():
        MEAN, STD = [0.4318, 0.3439, 0.32], [0.2418, 0.2133, 0.2183]
        return transforms.Normalize(mean=MEAN, std=STD)

    @staticmethod
    def get_denormalization_transform():
        MEAN, STD = [0.4318, 0.3439, 0.32], [0.2418, 0.2133, 0.2183]
        transform = DeNormalize(MEAN, STD)
        return transform
