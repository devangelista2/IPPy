import astra
import numpy as np
import torch


class Operator:
    r"""
    Defines the abstract Operator that will be subclassed for any specific case. It acts on standardized PyTorch tensors, i.e.
    tensors with shape (N, c, nx, ny), where N is the batch size, c is the number of channels in the data, nx and ny are the
    data spatial resolution. Each operator acts in parallel over the N elements of the batch. The input tensors are also assumed
    to be normalized in [0, 1]
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Compute y = Kx on the first element x
        y = self._matvec(x[0])

        # In case there are multiple x, compute y = Kx on all of them
        if x.shape[0] > 1:
            for i in range(1, x.shape[0]):
                y = torch.cat((y, self._matvec(x[i])))
        return y

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        return self.__call__(x)

    def T(self, y: torch.Tensor) -> torch.Tensor:
        # Compute x = K^Tx on the first element y
        x = self._adjoint(y[0])

        # In case there are multiple y, compute x = K^Ty on all of them
        if y.shape[0] > 1:
            for i in range(1, y.shape[0]):
                x = torch.cat((x, self._matvec(y[i])))
        return x


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
        y = self.proj @ x.flatten().numpy()
        return torch.tensor(y.reshape((1, 1, self.mx, self.my)))

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        x = self.proj.T @ y.flatten().numpy()
        return torch.tensor(x.reshape((1, 1, self.nx, self.ny)))

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
        c, nx, ny = x.shape
        D_h = torch.diff(x, n=1, dim=1, prepend=torch.zeros((c, 1, ny))).unsqueeze(0)
        D_v = torch.diff(x, n=1, dim=2, prepend=torch.zeros((c, nx, 1))).unsqueeze(0)

        return torch.cat((D_h, D_v), dim=1)

    def _adjoint(self, y: torch.Tensor) -> torch.Tensor:
        c, nx, ny = y.shape

        D_h = y[0, :, :]
        D_v = y[1, :, :]

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
