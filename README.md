# Inverse Problems with Python: IPPy

This README.md file is a work in progress. 

## Requirements
IPPy is built upon a few commonly used libraries for tensor manipulation, linear algebra, Computed Tomography, neural networks and visualization. Here is a list of the libraries you need to install to make IPPy run smoothly:

- `numpy`
- `torch`
- `torchvision`
- `numba`
- `astra-toolbox`
- `scikit-image`
- `PIL`
- `matplotlib`

Moreover, it is **required** to have access to a cuda GPU, both for training neural network models and for fast Computed Tomography simulations. In particular, some `Astra-toolbox` operators won't work if CUDA is not available.

## Data
You can use you own data to test. We provide a few example dataset to play with IPPy, which is also required to run the examples on the `examples/` folder, namely a modified version of the Mayo's dataset, and the COULE dataset:

- The COULE dataset (available on Kaggle at https://www.kaggle.com/datasets/loiboresearchgroup/coule-dataset) consists of 430 grey-scale images of dimension $256 \times 256$ representing ellipses of different contrast levels and small, high-contrasted dots, that imitates the human body structure, divided in 400 training images and 30 test images. More informations about the structure of the dataset is available at the link above.
- Mayo's Clinic Dataset (https://cdas.cancer.gov/datasets/mayo/) consists of 3305 grey-scale images of dimension $512 \times 512$, representing real anonymized CT reconstructions of human lungs from 10 patients, available at: https://drive.google.com/drive/folders/13BEiz6t57qSbwBpCtfqllmYTLmkhQeFE?usp=share_link.