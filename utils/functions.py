import numba, numpy as np
from numba import *
from numba.experimental import jitclass
from functools import partial

class Activations:
    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def sigmoid(x, deriv=False):
        if deriv:
            return x * (1 - x)

        return ( 1 / ( 1 + np.exp(-x) ) )

    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def tanh(x, deriv=False):
        if deriv:
            return (1 - x ** 2)

        return np.tanh(x)

    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def relu(x, deriv=False):

        negative_slope = 10 ** -9

        if deriv:
            return 1 * (x > 0) + (negative_slope * (x < 0))

        return 1 * x * (x > 0) + (negative_slope * x * (x < 0))

    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def crelu(x, deriv=False):
        if deriv:
            return (1 * ((x > 0) & (x < 1))).astype(np.float64)

        return x * ((x > 0) & (x < 1)) + (1 * (x >= 1))

    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def softmax(x, deriv=False):

        if deriv:
            softmax_output = np.exp(x) / np.sum(np.exp(x))
            return softmax_output * (1 - softmax_output)

        e_x = np.exp(x)

        return e_x / e_x.sum()

class Loss:
    @staticmethod
    @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def cross_entropy(outputs, expected_outputs, deriv=False):
        if deriv:
            return (outputs - expected_outputs) / outputs.shape[0]

        epsilon = 1e-12

        outputs = np.clip(outputs, epsilon, 1.0 - epsilon)

        return -(expected_outputs * np.log(outputs + epsilon))
        
    @staticmethod
    @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def mse(outputs, expected_outputs, deriv=False):
        if deriv:
            return 2 * (outputs - expected_outputs)
            
        return (outputs - expected_outputs) ** 2

    @staticmethod
    @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def smooth_l1_loss(outputs, expected_outputs, deriv=False):
        delta = 1.0

        if deriv:
            diff = outputs - expected_outputs
            mask = np.abs(diff) <= delta
            return np.where(mask, diff, np.sign(diff) * delta) / outputs.shape[0]
        else:
            diff = outputs - expected_outputs
            return np.where(np.abs(diff) <= delta, 0.5 * diff ** 2, delta * (np.abs(diff) - 0.5 * delta)) / outputs.shape[0]