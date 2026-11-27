from datasets.seq_cddb import SequentialCDDBHard
import torchvision.transforms as transforms
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode

class SequentialCDDB_san(SequentialCDDBHard):
    NAME = "seq-cddb-san"
    SETTING = "class-il"
    N_TASKS = 5
    MEAN, STD = [0.4321, 0.4348, 0.379], [0.253, 0.2394, 0.2482]
    TRAIN_PATH = "data/cddb-hard/san/train"
    TEST_PATH = "data/cddb-hard/san/val"
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
        MEAN, STD = [0.4321, 0.4348, 0.379], [0.253, 0.2394, 0.2482]
        transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        return transform
    @staticmethod
    def get_normalization_transform():
        MEAN, STD = [0.4321, 0.4348, 0.379], [0.253, 0.2394, 0.2482]
        return transforms.Normalize(mean=MEAN, std=STD)

    @staticmethod
    def get_denormalization_transform():
        MEAN, STD = [0.4321, 0.4348, 0.379], [0.253, 0.2394, 0.2482]
        transform = DeNormalize(MEAN, STD)
        return transform