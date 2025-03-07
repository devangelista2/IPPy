import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image


def create_path_if_not_exists(path: str) -> None:
    r"""
    Check if the path exists. If this is not the case, it creates the required folders.

    :param str path: The path to be checked and created.
    """
    if not os.path.isdir(path):
        os.makedirs(path)


def get_device() -> None:
    r"""
    Return the best possible device. In particular, if "cuda" is available, it returns
    "cuda". If "mps" is avabilable, it returns "mps". Otherwise, it returns "cpu".
    """
    try:
        if torch.mps.is_available():
            return "mps"
    except:
        pass

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalize(x: torch.Tensor) -> torch.Tensor:
    r"""
    Given an array x, returns its normalized version (i.e. the linear projection into [0, 1]).

    :param torch.Tensor x: The pytorch tensor to be normalized.
    """
    return (x - x.min()) / (x.max() - x.min())


def load_image(path: str) -> torch.Tensor:
    r"""
    Load a .png gray-scale image from path, and converts it to a tensor of shape (1, 1, nx, ny), normalized in [0, 1] range.

    :param str path: The path of the gray-scale image that has to be loaded.
    """
    x = torch.tensor(np.array(Image.open(path).convert("L"))).unsqueeze(0).unsqueeze(1)
    return normalize(x)


def save_image(x: torch.Tensor, save_path: str) -> None:
    r"""
    Given a standardized PyTorch tensor x as input with shape (1, 1, nx, ny), converts it to a PIL image and saves it to
    the given path.

    :param torch.Tensor x: standardized PyTorch tensor with shape (1, 1, nx, ny) to be saved.
    :param str save_path: the path to which x has to be saved.
    """
    # Convert to PIL Image
    x = to_pil_image(x[0, 0])

    # Save
    x.save(save_path)


def gaussian_noise(y: torch.Tensor, noise_level: str) -> torch.Tensor:
    r"""
    Returns a data-dependent sample of gaussian noise "e", with norm equal ||e|| = noise_level * || y ||.

    :param torch.Tensor y: The corrupted data y = Kx.
    :param str noise_level: The noise level.
    """
    e = torch.randn_like(y, device=y.device)
    return e / torch.norm(e) * torch.norm(y) * noise_level


def show(
    x: list[torch.Tensor] | torch.Tensor,
    title: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    r"""
    Visualize a list of pytorch arrays of shape (1, 1, nx, ny), representing gray-scale images.

    :param list[torch.Tensor] | torch.Tensor x: The tensor to be shown, or a list of tensors to be shown.
    :param list[str] title: If given, add the title to each corresponding image to be shown.
    :param str save_path: If given, saves the image to the given path.
    """
    if isinstance(x, list):
        N = len(x)

        for i in range(N):
            plt.subplot(1, N, i + 1)
            plt.imshow(x[i][0, 0], cmap="gray")
            plt.axis("off")
            if title is not None:
                plt.title(title[i])

                if save_path is not None:
                    plt.imsave(f"{save_path}/{title[i]}.png")

        plt.show()

    else:
        plt.imshow(x[0, 0], cmap="gray")
        plt.axis("off")
        if title is not None:
            plt.title(title)

            if save_path is not None:
                plt.imsave(f"{save_path}/{title}.png")
        plt.show()


def formatted_time(start_time: float) -> str:
    r"""
    Given a starting time, computes the difference between the actual time and the starting time, and returns a nice string
    representation of time, in the format %H:%M:%S.

    :param float start_time: The starting time.
    """
    total_time = time.time() - start_time

    # Convert elapsed time to hours, minutes, and seconds
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)

    # Format using an f-string with %H:%M:%S style
    formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    return formatted_time
