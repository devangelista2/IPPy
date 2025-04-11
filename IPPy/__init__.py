from . import solvers

from .nn import models
from .utilities import metrics

# List which algorithms are neural networks and which are variational solvers
direct = {
    "FBP": solvers.FBP,
    "Identity": solvers.Identity,
}

variational = {
    "ChambollePockTpVConstrained": solvers.ChambollePockTpVConstrained,
    "ChambollePockTpVUnconstrained": solvers.ChambollePockTpVUnconstrained,
    "CGLS": solvers.CGLS,
    "SGP": solvers.SGP,
}

neural_network = {
    "UNet": models.UNet,
}
