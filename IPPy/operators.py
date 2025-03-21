import astra
import numpy as np

import math
import torch
import torch.nn.functional as F


class OperatorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x):
        """Forward pass: applies op._matvec to each (c, h, w)"""
        device = x.device
        ctx.op = op  # Store the operator for backward pass

        # Initialize output tensor
        batch_size = x.shape[0]
        y = []

        # Apply the operator to each sample in the batch (over the batch dimension)
        for i in range(batch_size):
            y_i = op._matvec(x[i].unsqueeze(0))  # Apply to each (c, h, w) tensor
            y.append(y_i)

        # Stack the results back into a batch
        y = torch.cat(y, dim=0)
        ctx.save_for_backward(x)  # Save input for gradient computation
        return y.to(device)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: applies op._adjoint to each (c, h, w)"""
        op = ctx.op
        device = grad_output.device

        # Initialize gradient input tensor
        batch_size = grad_output.shape[0]
        grad_input = []

        # Apply the adjoint operator to each element in the batch
        for i in range(batch_size):
            grad_i = op._adjoint(
                grad_output[i].unsqueeze(0)
            )  # Apply adjoint to each (c, h, w)
            grad_input.append(grad_i)

        # Stack the gradients back into a batch
        grad_input = torch.cat(grad_input, dim=0)

        return None, grad_input.to(device)  # No gradient for `op`, only `x`


class Operator:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Applies operator using PyTorch autograd wrapper"""
        return OperatorFunction.apply(self, x)

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector multiplication"""
        return self.__call__(x)

    def T(self, y: torch.Tensor) -> torch.Tensor:
        """Transpose operator (adjoint)"""
        device = y.device
        # Apply adjoint to the batch
        return self._adjoint(y).to(device).requires_grad_(True)

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the operator to a single (c, h, w) tensor"""
        raise NotImplementedError

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint operator to a single (c, h, w) tensor"""
        raise NotImplementedError


class CTProjector(Operator):
    r"""
    Implements a CTProjector operator, given the image shape in the form of a tuple (nx, ny), the angular acquisitions in the form
    of a numpy array (theta_1, theta_2, ..., theta_n), the detector size and the type of geometry.
    """

    def __init__(
        self,
        img_shape: tuple[int],
        angles: np.array,
        det_size: int | None = None,
        geometry: str = "parallel",
    ) -> None:
        super().__init__()
        # Input setup
        self.nx, self.ny = img_shape

        # Geometry
        self.geometry = geometry

        # Projector setup
        if det_size is None:
            self.det_size = 2 * int(max(self.nx, self.ny))
        else:
            self.det_size = det_size
        self.angles = angles
        self.n_angles = len(angles)

        # Set sinogram shape
        self.mx, self.my = self.n_angles, self.det_size

        # Define projector
        self.proj = self._get_astra_projection_operator()
        self.shape = self.proj.shape

    # ASTRA Projector
    def _get_astra_projection_operator(self):
        # create geometries and projector
        if self.geometry == "parallel":
            proj_geom = astra.create_proj_geom(
                "parallel", 1.0, self.det_size, self.angles
            )
            vol_geom = astra.create_vol_geom(self.nx, self.ny)
            proj_id = astra.create_projector("linear", proj_geom, vol_geom)

        elif self.geometry == "fanflat":
            proj_geom = astra.create_proj_geom(
                "fanflat", 1.0, self.det_size, self.angles, 1800, 500
            )
            vol_geom = astra.create_vol_geom(self.nx, self.ny)
            proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

        else:
            print("Geometry (still) undefined.")
            return None

        return astra.OpTomo(proj_id)

    # On call, project
    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.cpu().numpy().flatten()
        y_np = self.proj @ x_np
        return torch.tensor(
            y_np.reshape((1, 1, self.mx, self.my)), device=x.device
        )  # Convert back to PyTorch

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """CT backprojection: Converts PyTorch -> NumPy, restores channel"""
        y_np = y.cpu().numpy().flatten()
        x_np = self.proj.T @ y_np.flatten()  # ASTRA expects (sinogram_dim, proj)
        return torch.tensor(x_np.reshape((1, 1, self.nx, self.ny)), device=y.device)

    # FBP
    def FBP(self, y: torch.Tensor) -> torch.Tensor:
        # Compute x = K^Tx on the first element y
        x = self.proj.reconstruct("FBP_CUDA", y[0].numpy().flatten())
        x = torch.tensor(x.reshape((1, 1, self.nx, self.ny)))

        # In case there are multiple y, compute x = K^Ty on all of them
        if y.shape[0] > 1:
            for i in range(1, y.shape[0]):
                x_tmp = self.proj.reconstruct("FBP_CUDA", y[i].numpy().flatten())
                x_tmp = torch.tensor(x_tmp.reshape((1, 1, self.nx, self.ny)))
                x = torch.cat((x, x_tmp))
        return x


class Blurring(Operator):
    def __init__(
        self,
        kernel: torch.Tensor = None,
        kernel_type: str = None,
        kernel_size: int = 3,
        kernel_variance: float = 1.0,
        motion_angle: float = 0.0,
    ):
        """
        Blurring operator using convolution.

        Parameters:
        - kernel (torch.Tensor, optional): Custom kernel for convolution. If `kernel_type` is provided, this is ignored.
        - kernel_type (str, optional): Type of kernel to use. Supports 'gaussian' and 'motion'.
        - kernel_size (int, optional): Size of the kernel (for 'gaussian' and 'motion'). Must be an odd integer.
        - kernel_variance (float, optional): Variance of the Gaussian kernel (only used if kernel_type='gaussian').
        - motion_angle (float, optional): Angle of motion blur in degrees (only used if kernel_type='motion').
        """
        super().__init__()

        if kernel_type is not None:
            if kernel_type == "gaussian":
                if kernel_size % 2 == 0:
                    raise ValueError("Kernel size must be an odd integer.")
                self.kernel = self._generate_gaussian_kernel(
                    kernel_size, kernel_variance
                )
            elif kernel_type == "motion":
                if kernel_size % 2 == 0:
                    raise ValueError("Kernel size must be an odd integer.")
                self.kernel = self._generate_motion_kernel(kernel_size, motion_angle)
            else:
                raise ValueError("kernel_type must be either 'gaussian' or 'motion'")
        elif kernel is None:
            raise ValueError("Either `kernel` or `kernel_type` must be provided.")
        else:
            self.kernel = kernel

        # Ensure kernel is a 4D tensor with shape (out_channels, in_channels, k, k)
        if len(self.kernel.shape) == 2:
            self.kernel = self.kernel.unsqueeze(0)

        if len(self.kernel.shape) == 3:
            # Meaning only the batch dimension in missing
            self.kernel = self.kernel.unsqueeze(0)

    def _generate_gaussian_kernel(
        self, kernel_size: int, kernel_variance: float
    ) -> torch.Tensor:
        """
        Generates a Gaussian kernel with the given size and variance.
        """
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * kernel_variance))
        kernel /= kernel.sum()  # Normalize kernel to sum to 1
        return kernel

    def _generate_motion_kernel(self, kernel_size: int, angle: float) -> torch.Tensor:
        """
        Generates a motion blur kernel that blurs in a linear direction.
        """
        kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        center = kernel_size // 2
        angle = math.radians(angle)

        # Compute motion blur direction
        dx, dy = math.cos(angle), math.sin(angle)
        for i in range(kernel_size):
            x = int(center + (i - center) * dx)
            y = int(center + (i - center) * dy)
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        kernel /= kernel.sum()  # Normalize to keep intensity unchanged
        return kernel

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the blurring operator (forward convolution).
        """
        blurred = F.conv2d(x, self.kernel, padding="same")  # Apply convolution
        return blurred

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Applies the adjoint operator, which in this case is also a convolution
        with a flipped kernel (assuming symmetric kernels like Gaussian).
        """
        flipped_kernel = torch.flip(self.kernel, dims=[2, 3])  # Flip spatial dimensions
        adjoint_result = F.conv2d(y, flipped_kernel, padding="same")
        return adjoint_result


class DownScaling(Operator):
    def __init__(self, downscale_factor: int, mode: str = "avg"):
        """
        Initializes the DownScaling operator.

        Parameters:
        - downscale_factor (int): The factor by which the input is downscaled.
        - mode (str): The type of downscaling, either "avg" (average pooling) or "naive" (removes odd indices).
        """
        super().__init__()
        self.downscale_factor = downscale_factor
        if mode not in ["avg", "naive"]:
            raise ValueError("mode must be either 'avg' or 'naive'")
        self.mode = mode

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the downscaling operator.
        """
        factor = self.downscale_factor
        if self.mode == "avg":
            y = F.avg_pool2d(x, factor, stride=factor)
        elif self.mode == "naive":
            y = x[..., ::factor, ::factor]  # Take every 'factor'-th element
        return y

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """
        Applies the transposed operator (true adjoint).
        """
        factor = self.downscale_factor
        if self.mode == "avg":
            # Spread values uniformly back
            y_upsampled = F.interpolate(y, scale_factor=factor, mode="nearest")
            x_out = y_upsampled / (factor**2)  # Normalize to preserve energy
        elif self.mode == "naive":
            # Create a zero-filled tensor of the original size
            N, C, H, W = y.shape
            H_up, W_up = H * factor, W * factor
            x_up = torch.zeros((N, C, H_up, W_up), device=y.device, dtype=y.dtype)
            x_up[..., ::factor, ::factor] = y  # Insert values at correct locations
            x_out = x_up
        return x_out


class Gradient(Operator):
    r"""
    Implements the Gradient operator, acting on standardized Pytorch tensors of shape (N, 1, nx, ny) and returning a tensor of
    shape (N, 2, nx, ny), where the first channel contains horizontal derivatives, while the second channel contains vertical
    derivatives.
    """

    def __init__(self, img_shape: tuple[int]) -> None:
        super().__init__()

        self.nx, self.ny = img_shape
        self.mx, self.my = img_shape

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        N, c, nx, ny = x.shape
        D_h = torch.diff(x, n=1, dim=1, prepend=torch.zeros((N, c, 1, ny))).unsqueeze(0)
        D_v = torch.diff(x, n=1, dim=2, prepend=torch.zeros((N, c, nx, 1))).unsqueeze(0)

        return torch.cat((D_h, D_v), dim=1)

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        N, c, nx, ny = y.shape

        D_h = y[0, 0, :, :]
        D_v = y[0, 1, :, :]

        D_h_T = (
            torch.flipud(
                torch.diff(torch.flipud(D_h), n=1, dim=0, prepend=torch.zeros((1, ny)))
            )
            .unsqueeze(0)
            .unsqueeze(1)
        )
        D_v_T = (
            torch.fliplr(
                torch.diff(torch.fliplr(D_v), n=1, dim=1, prepend=torch.zeros((nx, 1)))
            )
            .unsqueeze(0)
            .unsqueeze(1)
        )

        return D_h_T + D_v_T
