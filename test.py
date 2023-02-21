import numpy as np
import matplotlib.pyplot as plt

from tomography.data import generate_phantom, generate_COULE
import operators, solvers



x = generate_phantom.phantom(p_type='modified shepp-logan')
m, n = x.shape

A = operators.CTProjector(m, n, np.linspace(0, 150, 50), 256)

y = A(x)
np.random.seed(42)
y = y + np.random.normal(0, 1, y.shape) * 1

CP_TV = solvers.ChambollePockTpV(A)
x_rec = CP_TV(y, epsilon=1e-5*np.max(y)*np.sqrt(m), lmbda=1e-1, maxiter=15, x_true=x.flatten(), p=0.1)

plt.subplot(1, 3, 1)
plt.imshow(x, cmap='gray')

plt.subplot(1, 3, 2)
plt.imshow(A.FBP(y).reshape((256, 256)), cmap='gray')

plt.subplot(1, 3, 3)
plt.imshow(x_rec.reshape((256, 256)), cmap='gray')
plt.show()