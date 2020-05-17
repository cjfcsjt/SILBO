import math
import numpy as np
# import matlab.engine
import torch
from test_functions.mnist_weight import mnist_weight

# All functions are defined in such a way that have global maximums,
# if a function originally has a minimum, the final objective value is multiplied by -1

class TestFunction:
    def evaluate(self,x):
        pass

class Rosenbrock(TestFunction):
    def __init__(self, act_var, noise_var=0):
        self.range=np.array([[-2,2],
                             [-2,2]])
        self.act_var=act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i,1] - self.range[i,0]) / 2 + (
                    self.range[i,1] + self.range[i,0]) / 2
        return x_copy

    def evaluate_true(self,x):
        # Calculating the output
        scaled_x=self.scale_domain(x)
        test = self.act_var[0]
        for i in scaled_x:
            test1 = i[self.act_var[0]]
        f = [[0]]
        f[0] = [-(math.pow(1 - i[self.act_var[0]], 2) + 100 * math.pow(i[self.act_var[1]] - math.pow(i[self.act_var[0]], 2), 2)) for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))

class Branin(TestFunction):
    def __init__(self, act_var, noise_var=0):
        self.range=np.array([[-5,10],
                             [0,15]])
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self,x):
        scaled_x=self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [-((i[self.act_var[1]] - (5.1 / (4 * math.pi ** 2)) * i[self.act_var[0]] ** 2 + i[self.act_var[0]] * 5 / math.pi - 6) ** 2 + 10 * (
                1 - 1 / (8 * math.pi)) * np.cos(i[self.act_var[0]]) + 10) for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))

class Hartmann6(TestFunction):
    def __init__(self, act_var, noise_var=0):
        self.range = np.array([[0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1]])
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self,x):
        # Calculating the output
        #Created on 08.09.2016
        # @author: Stefan Falkner
        alpha = [1.00, 1.20, 3.00, 3.20]
        A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                      [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                      [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                      [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
        P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])
        scaled_x = self.scale_domain(x)
        n=len(scaled_x)
        external_sum = np.zeros((n,1))
        for r in range(n):
            for i in range(4):
                internal_sum = 0
                for j in range(6):
                    internal_sum = internal_sum + A[i, j] * (scaled_x[r, self.act_var[j]] - P[i, j]) ** 2
                external_sum[r] = external_sum[r] + alpha[i] * np.exp(-internal_sum)
        return external_sum

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))

class StybTang(TestFunction):
    def __init__(self, act_var, noise_var=0):
        D = len(act_var)
        a = np.ones((D, 2))
        a = a * 5
        a[:, 0] = a[:, 0] * -1
        self.range = a
        self.act_var=act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i,1] - self.range[i,0]) / 2 + (
                    self.range[i,1] + self.range[i,0]) / 2
        return x_copy

    def evaluate_true(self,x):
        # Calculating the output
        scaled_x=self.scale_domain(x)
        f = [-0.5 * np.sum(np.power(scaled_x, 4) - 16 * np.power(scaled_x, 2) + 5 * scaled_x, axis=1)]
        f = np.transpose(f)
        return f

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))

class bell(TestFunction):
    def __init__(self, act_var, noise_var=0):
        self.range=np.array([[-4.5,4.5],
                             [-4.5,4.5]])
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self,x):
        scaled_x=self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [-(
            (1.5-i[self.act_var[0]]+i[self.act_var[0]]*i[self.act_var[1]])**2
            +(2.25-i[self.act_var[0]]+i[self.act_var[0]]*i[self.act_var[1]]**2)**2
            +(2.625-i[self.act_var[0]]+i[self.act_var[0]]*i[self.act_var[1]]**3)**2
        ) for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))




class MNIST(object):
    def __init__(self, act_var):
        D = len(act_var)
        a = np.ones((D, 2))
        a = a * 1
        a[:, 0] = a[:, 0] * -1
        self.range = a
        self.act_var = act_var


    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self,x):
        scaled_x = self.scale_domain(x)
        if len(scaled_x.shape) == 1:
            scaled_x = scaled_x.reshape((1, scaled_x.shape[0]))
        n = len(scaled_x)
        res = np.zeros((n, 1))
        for i in range(n):
            x = scaled_x[i]
            x = torch.from_numpy(x).type(torch.FloatTensor)
            res[i] = -mnist_weight(x)
        return res

    def evaluate(self, x):
        return self.evaluate_true(x)


class Quadratic(TestFunction):
    def __init__(self, act_var=None, noise_var=0):
        self.range = np.array([[-1, 1],
                               [-1, 1]])
        if act_var is None:
            self.act_var = np.arange(self.range.shape[0])
        else:
            self.act_var = act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self,x):
        scaled_x = self.scale_domain(x)
        f = [[0]]
        f[0] = [-((i[self.act_var[0]]-1)**2+(i[self.act_var[1]]-1)**2) for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))

class camel3(TestFunction):
    def __init__(self, act_var, noise_var=0):
        self.range=np.array([[-5,5],
                             [-5,5]])
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self,x):
        scaled_x=self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [2 * i[self.act_var[0]] ** 2 +  -1.05 * i[self.act_var[0]] ** 4 + i[self.act_var[0]] ** 6 / 6 +
                i[self.act_var[0]] * i[self.act_var[1]] +  i[self.act_var[1]] ** 2 for i in scaled_x]


        f = np.transpose(f)
        return f

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))

class goldstein_price(TestFunction):
    def __init__(self, act_var, noise_var=0):
        self.range=np.array([[-2,2],
                             [-2,2]])
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self,x):
        scaled_x=self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [ -(1+(i[self.act_var[0]]+i[self.act_var[0]]+1)**2*(19-14*i[self.act_var[0]]+3*i[self.act_var[0]]**2-14*i[self.act_var[0]]+
                6*i[self.act_var[0]]*i[self.act_var[0]]+3*i[self.act_var[0]]**2))*(30+(2*i[self.act_var[0]]-3*i[self.act_var[0]])**2*
                  (18-32*i[self.act_var[0]]+12*i[self.act_var[0]]**2
                        +48*i[self.act_var[0]]-36*i[self.act_var[0]]*i[self.act_var[0]]+27*i[self.act_var[0]]**2))for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))

class colville(TestFunction):
    def __init__(self, act_var, noise_var=0):
        self.range=np.array([[-10,10],
                             [-10,10],
                             [-10, 10],
                             [-10, 10]])
        self.act_var = act_var
        self.var = noise_var

    def scale_domain(self,x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                        self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self,x):
        scaled_x=self.scale_domain(x)
        # Calculating the output
        f = [[0]]
        f[0] = [ -(100 * (i[self.act_var[0]]**2-i[self.act_var[1]])**2+(i[self.act_var[0]]-1)**2+(i[self.act_var[2]]-1)**2+90 * (i[self.act_var[2]]**2-i[self.act_var[4]])**2+
                 10.1 * ((i[self.act_var[1]]-1)**2 + (i[self.act_var[3]]-1)**2)+19.8*(i[self.act_var[1]]-1)*(i[self.act_var[3]]-1))
                 for i in scaled_x]
        f = np.transpose(f)
        return f

    def evaluate(self, x):
        scaled_x = self.scale_domain(x)
        n = len(scaled_x)
        return self.evaluate_true(x) + np.random.normal(0,self.var,(n,1))
