# Required a funcion A(x) that takes as input x (dimension n**2) and returns A@x.
# Required A to have a transpose .T operator.
import numpy as np
import matplotlib.pyplot as plt

from . import operators, metrics, reconstructors


class CGLS:
    def __init__(self, A):
        self.A = A
    
    def __call__(self, b, x0, x_true=None, kmax=100, tolf=1e-6, tolx=1e-6, info=False):
        d  = b
        r0 = self.A.T(b)
        p = r0
        t = self.A @ p
        
        x = x0
        r = r0 
        k = 0

        if x_true is not None:
            err_vec = np.zeros((kmax, 1))
            err_vec[k] = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)

            condition = np.linalg.norm(r) > tolf and err_vec[k] > tolx and k < kmax -1
        else:
            condition = np.linalg.norm(r) > tolf and k < kmax -1
        while condition:
            x0 = x

            alpha = np.linalg.norm(r0, 2)**2 / np.linalg.norm(t, 2)**2
            x = x0 + alpha * p
            d = d - alpha * t
            r = self.A.T(d)
            beta = np.linalg.norm(r, 2)**2 / np.linalg.norm(r0, 2)**2
            p = r + beta * p
            t = self.A @ p
            k = k + 1

            r0 = r

            if x_true is not None:
                err_vec[k] = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
                condition = np.linalg.norm(r) > tolf and err_vec[k] > tolx and k < kmax -1

            else:
                condition = np.linalg.norm(r) > tolf and k < kmax -1

        if x_true is not None:        
            err_vec = err_vec[:k+1]

            if info:
                return x, err_vec
        return x

class ChambollePockTpV:
    def __init__(self, A):
        self.A = A

        self.m, self.n = A.shape
        
        # Generate Gradient operators
        self.grad = operators.myGradient(1, (int(np.sqrt(self.n)), int(np.sqrt(self.n))))

        self.m, self.n = A.shape

    def __call__(self, b, epsilon, lmbda, x_true=None, starting_point=None, eta=2e-3, maxiter=100, p=1):
        # Compute the approximation to || A ||_2
        nu = np.sqrt(self.power_method(self.A, num_iterations=10) / self.power_method(self.grad, num_iterations=10))

        # Generate concatenate operator
        K = operators.ConcatenateOperator(self.A, self.grad)

        Gamma = np.sqrt(self.power_method(K, num_iterations=10))

        # Compute the parameters given Gamma
        tau = 1 / Gamma
        sigma = 1 / Gamma
        theta = 1
        
        # Iteration counter
        k = 0

        # Initialization
        if starting_point is None:
            x = np.zeros((self.n, 1))
        else:
            x = starting_point
        y = np.zeros((self.m, 1))
        w = np.zeros((2 * self.n, 1))

        xx = x

        # Initialize errors
        rel_err = np.zeros((maxiter+1, 1))
        residues = np.zeros((maxiter+1, 1))

        # Stopping conditions
        con = True
        while con and (k < maxiter):
            # Update y
            yy = y + sigma * np.expand_dims(self.A(xx) - b, -1)
            y = max(np.linalg.norm(yy) - (sigma*epsilon), 0) * yy / np.linalg.norm(yy)

            # Compute the magnitude of the gradient
            grad_x = self.grad(xx)
            grad_mag = np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])

            # Compute the reweighting factor
            W = np.expand_dims(np.power(np.sqrt(eta**2 + grad_mag) / eta, p-1), -1)
            WW = np.concatenate((W, W), axis=0)

            # Update w
            x_grad = np.expand_dims(self.grad(xx), -1)
            ww = w + sigma * x_grad

            abs_ww = np.zeros((self.n, 1))
            # for i in range(self.n):
            #     abs_ww[i] = ww[i]**2 + ww[i+self.n]**2
            abs_ww = np.square(ww[:self.n]) + np.square(ww[self.n:])
            abs_ww = np.concatenate((abs_ww, abs_ww), axis=0)
            
            lmbda_vec_over_nu = lmbda * WW / nu
            w = lmbda_vec_over_nu * ww / np.maximum(lmbda_vec_over_nu, abs_ww)

            # Save the value of x
            xtmp = x

            # Update x
            x = xtmp - tau * (np.expand_dims(self.A.T(y), -1) + nu * np.expand_dims(self.grad.T(w), -1))

            # Project x to (x>0)
            x[x<0] = 0

            # Compte signed x
            xx = x + theta * (x - xtmp)

            # Compute relative error
            if x_true is not None:
                rel_err[k] = np.linalg.norm(xx.flatten() - x_true.flatten()) / np.linalg.norm(x_true.flatten())

            # Compute the magnitude of the gradient of the actual iterate
            grad_x = self.grad(xx)
            grad_mag = np.expand_dims(np.sqrt(np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])), -1)

            # Compute the value of TpV by reweighting
            ftpv = np.sum(np.abs(W * grad_mag))
            res = np.linalg.norm(self.A(xx) - b, 2)**2
            residues[k] = 0.5 * res + lmbda * ftpv

            # Stopping criteria
            c = np.sqrt(res) / (np.max(b) * np.sqrt(self.m))
            if (c>= 9e-6) and (c<=1.1e-5):
                con = False

            # Update k
            k = k + 1
            # print(k, rel_err[k-1])

        return x
    
    def power_method(self, A, num_iterations: int):
        b_k = np.random.rand(A.shape[1])

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = A.T(A(b_k))

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm
    
    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = np.expand_dims(self.grad(x), -1)
        grad_mag = np.square(grad_x[:len(grad_x)//2]) + np.square(grad_x[len(grad_x)//2:])
        W = np.power(np.sqrt(eta**2 + grad_mag) / eta, p-1)

        ftpv = np.sum(np.abs(W * np.sqrt(grad_mag)))
        return 0.5 * np.linalg.norm(self.A(x) - y, 2)**2 + lmbda * ftpv