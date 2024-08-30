import glob
import os

# Ignore warnings
import warnings

from torch.utils.data import Dataset
from torchvision.transforms import Resize

from ._utilities import *

# Disable warnings
warnings.filterwarnings("ignore")


class ImageDataset(Dataset):
    """
    Implements a Dataset subclass that reads .png gray-scale images from a folder of data and converts it
    to a standardized pytorch Tensor with the given shape. Note that the data_path should be the path to a folder, containing
    multiple folders (one per patient), each containing the .png files.

    :param str data_path: (Relative) path to the dataset.
    :param int data_shape: The value of nx = ny. If different to the true data shape, each tensor get reshaped to the required shape.
    """

    def __init__(self, data_path: str, data_shape: int | None = None) -> None:
        self.data_path = data_path
        self.img_name_list = sorted(
            glob.glob(os.path.join(self.data_path, "*", "*.png"))
        )

        self.data_shape = data_shape

    def __len__(self) -> int:
        return len(self.img_name_list)

    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            img = load_image(self.img_name_list[index])
        elif isinstance(index, slice):
            img = load_image(self.img_name_list[0])
            for i in range(index.start + 1, index.stop):
                img = torch.cat((img, load_image(self.img_name_list[i])), dim=0)

        if self.data_shape is not None:
            img = Resize(self.data_shape)(img)

        return img, self.get_name(index)

    def get_name(self, index: int) -> str:
        r"""
        The full path to the required index.
        """
        return self.img_name_list[index]


class TrainDataset(Dataset):
    """
    Implements a Dataset subclass that reads .png gray-scale images from two folders of data and converts it
    to a standardized pytorch Tensor with the given shape. Note that both in_path and out_path should be
    the path to a folder, containing multiple folders (one per patient), each containing the .png files. The structure of
    in_path and out_path has to be same.

    :param str in_path: (Relative) path to the input dataset.
    :param str out_path: (Relative) path to the target dataset.
    :param int data_shape: The value of nx = ny. If different to the true data shape, each tensor get reshaped to the required shape.
    """

    def __init__(
        self, in_path: str, out_path: str, data_shape: int | None = None
    ) -> None:
        self.in_path = in_path
        self.out_path = out_path

        self.in_name_list = sorted(glob.glob(os.path.join(self.in_path, "*", "*.png")))
        self.out_name_list = sorted(
            glob.glob(os.path.join(self.out_path, "*", "*.png"))
        )
        assert len(self.in_name_list) == len(self.out_name_list)

        self.data_shape = data_shape

    def __len__(self) -> int:
        return len(self.in_name_list)

    def __getitem__(self, index: int | slice):
        if isinstance(index, int):
            x = load_image(self.in_name_list[index])[0]
            y = load_image(self.out_name_list[index])[0]
        elif isinstance(index, slice):
            x = load_image(self.in_name_list[0])
            y = load_image(self.out_name_list[0])
            for i in range(index.start + 1, index.stop):
                x = torch.cat((x, load_image(self.in_name_list[i])), dim=0)
                y = torch.cat((y, load_image(self.out_name_list[i])), dim=0)

        if self.data_shape is not None:
            x = Resize(self.data_shape)(x)
            y = Resize(self.data_shape)(y)

        return x, y
