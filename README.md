# IPPy - Inverse Problems with Python
`IPPy` is a Python collection of tools to solve Inverse Problems. It follows the notations from the paper *To be or not to be stable, that is the question: understanding neural networks for inverse problems*, where a problem is solved by using a **reconstructor**, the main object of `IPPy`.

## Installation
To install `IPPy`, clone this GitHub repository locally by running

```
git clone https://github.com/devangelista2/IPPy.git
```

## Mathematical background and notation

### Introduction to Inverse Problems
Here I summarize the mathematics of Inverse Problems required to understand the basics of `IPPy`. Given a continuous, linear **operator** $K \in \mathbb{R}^{m \times n}$ and an unknown image $x \in \mathbb{R}^n$, consider the linear system

$$
    y^\delta = Kx + e
$$

where $e$ represents the noise corruption with $|| e || \leq \delta$, while $y^\delta \in \mathbb{R}^m$ represents the observable noisy image, altered by the action of the operator $K$ and the noise. The problem of recovering an *approximation* of $x$ given $y^\delta$ is called **inverse problem**. 

If $\delta = 0$, $m = n$ and $K$ is invertible, then the problem above can be easily solved as

$$
    \tilde{x} = K^{-1}y^0.
$$

Clearly, in real-life applications, $\delta$ will usually be strictly greater than 0 and $K$ will tipically be non-invertible or, at least, **ill-conditioned**, meaning that the solution will amplifies the noise in input unboundedly.

When $K$ is not invertible, it is typical to consider $\tilde{x}$ to be the solution to the least-square problem

$$
    x^\dagger = \arg\min_{x \in \mathbb{R}^n} \frac{1}{2} || Kx - y^\delta ||_2^2,
$$

which can be compactly written as

$$
    x^\dagger = K^\dagger y^\delta
$$

where $K^\dagger$ is the Moore-Penrose pseudo-inverse of $K$.

When $K$ is ill-conditioned, a solution as $x^\dagger$ will be unsatisfactory due to the amplification of noise caused by $K^\dagger$. To solve this issue, we introduced the main concept of `IPPy`: the **reconstructors**.

A **reconstructor** is defined as any continuous function $\Psi: \mathbb{R}^m \to \mathbb{R}^n$, acting on $y^\delta$, trying to map it to an approximation of $x$ such that $y^\delta = Kx + e$. A reconstructor $\Psi$ is associated to two main property:

* **Accuracy:** Quantifies the ability of $\Psi$ in solving a noise-less Inverse Problem (i.e. $\delta = 0$). It is defined to be $\eta^{-1}$ with 
$$
    \eta = \sup_{x \in \mathbb{R}^n} || \Psi(y^0) - x ||_2.
$$
* **Stability:** Quantifies how much $\Psi$ amplifies noise with norm less than a given $\delta > 0$ in input. It is defined as
$$
    C^\delta_\Psi = \sup_{\substack{x \in \mathbb{R}^n \\ ||e||_2 \leq \delta}} \frac{|| \Psi(y^\delta) - x ||_2 - \eta}{||e||_2}.
$$

The accuracy and the stability are related by a trade-off, meaning that increasing the stability will necessarily reduce the accuracy, and vice-versa. More details about the theory of reconstructors can be found in the paper.

### Reconstructors

Here, we consider two main kind of reconstructors: the **variational reconstructor** and the **neural network reconstructor**.

The **Variational reconstructor** is defined as the function mapping $y^\delta$ to the solution of a regularized least square optimization problem. In particual we consider Tikhonov-regularized reconstructors. Given a matrix $L \in \mathbb{R}^{d \times n}$ and $\lambda > 0$, we define the Tikhonov-regularized reconstructor with matrix $L$ and parameter $\lambda>0$ as

$$
    \Psi^{\lambda, L}(y^\delta) = \arg\min_{x \in \mathbb{R}^n} \frac{1}{2} || Kx - y^\delta ||_2^2 + \frac{\lambda}{2} || Lx ||_2^2.
$$

Clearly, for any matrix $L$, if $\lambda = 0$ then

$$
    \Psi^{0, L}(y^\delta) = K^\dagger y^\delta
$$

is the reconstructor defined at the beginning of the section. Since the Tikhonov-regularized reconstrutor is a modified least-square problem, its solution can be expressed in normal-equation form as

$$
    \Psi^{\lambda, L}(y^\delta) = (K^T K + \lambda L^T L)^{-1}K^T y^\delta.
$$

To solve this problem, we implemented `CGLS`, a powerful iterative algorithm used to solve least-squares problem efficiently. Clearly, the quality of the reconstruction strongly depends on the choice of $\lambda$.

The **Neural Network reconstructor** is defined to be just a neural network $\Psi_\theta \in \mathcal{F}_\theta := \{ \Psi_\theta : \R^m \to \R^n; \theta \in \mathbb{R}^s \}$, trained on the dataset $\mathcal{S} = \{ (y^\delta_i, x_i) \}_{i=1}^N$ to solve the inverse problem above. By construction, $\Psi_\theta$ will be highly accurate but potentially very unstable.

### Stabilizers

To improve the stability of neural networks, we introduced the concept of **stabilizers**, which are basically pre-processing operators designed to remove noise as input before processing it through the network. We have introduced to kind of stabilizers: a **Gaussian filter**, cutting out the high frequencies of the input and thus reducing the noise, and an **iterative stabilizer**, that consider a small number of iteration of `CGLS` algorithm applied to the variational reconstructor defined above. If $\lambda > 0$ and the number $k$ of iterations are big enough, then the obtained stabilizer will improve the stability of the input to a large amount.

## Structure of the library
Here I discuss how each Python file in the main package of `IPPy` should be used.

* `reconstructors.py` implements the class that defines the reconstructor. In particular, we defined the `VariationalReconstructor`, implementing the regularized least-square problem given an algorithm to solve it (from the `solvers.py` module).
* `operators.py` implements a long list of linear operators, defining the forward problem $K$. It ranges from `ConvolutionalOperator`, defining the blurring operator, to `CTOperator`, defining the Computed Tomography (CT) projector (from AstraToolbox). Also, the `IdentityOperator` and the `GradientOperator` are introduced (the names are self-explicative). Finally, the `MatrixOperator` converts a numpy matrix to an `IPPy` operator, and the `ConcatenateOperator` can be used to concatenate two `IPPy` operators together.
* `stabilizers.py` contains the classes defining the stabilizers introduced above. In particular, three possible choices for the stabilizer can be done: the `PhiIdentity` (implementing the identity function), the `GaussianFilter` and the `Tik_CGLS_stabilizer`.
* `solvers.py` contains the implementation of iterative algorithms to solve regularized least squares problem. In particular, an implementation for `CGLS` can be found, together with a `ChambollePockTV`, to solve Total-Variation regularized problems by using the Chambolle Pock algorithm.
* `metrics.py` implements some metrics used to evaluate the quality of the reconstruction, such as relative error, PSNR and SSIM.
* `utils.py` implements some utility functions.

### The sub-package `nn`
In `IPPy.nn` can be found the implementation of some state-of-the-art neural network models for image reconstructions. In particular, we implemented the UNet [1] model, the NAFNet [2] model and the SSNet [3] model. Each model can be invoked by the `get_MODELNAME` function, from the `models.py` module. Also, an implementation of the UNet3D can be found in the `models3d.py` module.

### References
[1] *U-Net: Convolutional Networks for Biomedical Image Segmentation*, Olaf Ronneberger, Philipp Fischer, Thomas Brox; 2015. </br> 
[2] *Simple Baselines for Image Restoration*, L. Chen; 2022. </br>
[3] *A green prospective for learned post-processing in sparse-view tomographic reconstruction*, E. Morotti, D. Evangelista, E. Loli Piccolomini, 2020. </br>

## Results (Coming soon)
Results are coming soon.

## BibTex citation
To cite our work, use:

```
@article{evangelista2022or,
  title={To be or not to be stable, that is the question: understanding neural networks for inverse problems},
  author={Evangelista, Davide and Nagy, James and Morotti, Elena and Piccolomini, Elena Loli},
  journal={arXiv preprint arXiv:2211.13692},
  year={2022}
}
```
