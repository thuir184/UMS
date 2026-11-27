from datasets.seq_cddb import SequentialCDDBHard
import torchvision.transforms as transforms
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode

class SequentialCDDB_biggan(SequentialCDDBHard):
    NAME = "seq-cddb-biggan"
    SETTING = "class-il"
    N_TASKS = 5
    
    MEAN, STD = [0.4693, 0.4509, 0.3954], [0.2711, 0.2611, 0.2747]
    TRAIN_PATH = "data/cddb-hard/biggan/train"
    TEST_PATH = "data/cddb-hard/biggan/val"
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
        MEAN, STD = [0.4693, 0.4509, 0.3954], [0.2711, 0.2611, 0.2747]
        transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        return transform
    @staticmethod
    def get_normalization_transform():
        MEAN, STD = [0.4693, 0.4509, 0.3954], [0.2711, 0.2611, 0.2747]
        return transforms.Normalize(mean=MEAN, std=STD)

    @staticmethod
    def get_denormalization_transform():
        MEAN, STD = [0.4693, 0.4509, 0.3954], [0.2711, 0.2611, 0.2747]
        transform = DeNormalize(MEAN, STD)
        return transform