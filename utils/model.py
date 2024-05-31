import scipy, skimage, numba, time, numpy as np
from numba import *
from numba.experimental import jitclass
from functools import partial
from typing import *

spec = [
    ('model', double[:, :, :]),
    ('heights', int32[:])
]
 
class Model:
    def __init__(self, model, heights, convolutional_model, convolutional_layers, convolutional_biases, hidden_function="tanh", output_function="softmax", cost_function="cross_entropy"):
        self.model = model
        self.heights = heights
        self.convolutional_model = convolutional_model
        self.convolutional_layers = convolutional_layers
        self.convolutional_biases = convolutional_biases
        self.hidden_layer_activation_function = getattr(self, "_"+hidden_function)
        self.output_layer_activation_function = getattr(self, "_"+output_function)
        self.cost_function = getattr(self, "_"+cost_function)
    
    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def _sigmoid(x, deriv=False):
        if deriv:
            return x * (1 - x)

        return ( 1 / ( 1 + np.exp(-x) ) )

    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def _tanh(x, deriv=False):
        if deriv:
            return (1 - x ** 2)

        return np.tanh(x)

    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def _relu(x, deriv=False):

        negative_slope = 10 ** -9

        if deriv:
            return 1 * (x > 0) + (negative_slope * (x < 0))

        return 1 * x * (x > 0) + (negative_slope * x * (x < 0))

    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def _crelu(x, deriv=False):
        if deriv:
            return (1 * ((x > 0) & (x < 1))).astype(np.float64)

        return x * ((x > 0) & (x < 1)) + (1 * (x >= 1))

    @staticmethod
    @numba.cfunc("float64[:](float64[:], optional(boolean))")
    def _softmax(x, deriv=False):

        if deriv:
            softmax_output = np.exp(x) / np.sum(np.exp(x))
            return softmax_output * (1 - softmax_output)

        e_x = np.exp(x)

        return e_x / e_x.sum()
            
    @staticmethod
    @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def _cross_entropy(outputs, expected_outputs, deriv=False):
        if deriv:
            return (outputs - expected_outputs) / outputs.shape[0]

        epsilon = 1e-12

        outputs = np.clip(outputs, epsilon, 1.0 - epsilon)

        return -(expected_outputs * np.log(outputs + epsilon))
        
    @staticmethod
    @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def _mse(outputs, expected_outputs, deriv=False):
        if deriv:
            return 2 * (outputs - expected_outputs)
            
        return (outputs - expected_outputs) ** 2

    @staticmethod
    @numba.cfunc("float64[:](float64[:], float64[:], optional(boolean))")
    def _smooth_l1_loss(outputs, expected_outputs, deriv=False):
        delta = 1.0

        if deriv:
            diff = outputs - expected_outputs
            mask = np.abs(diff) <= delta
            return np.where(mask, diff, np.sign(diff) * delta) / outputs.shape[0]
        else:
            diff = outputs - expected_outputs
            return np.where(np.abs(diff) <= delta, 0.5 * diff ** 2, delta * (np.abs(diff) - 0.5 * delta)) / outputs.shape[0]

    # @jit(forceobj=True, cache=False)
    def gradient(self, activations, expected_output, weight_decay = 0):
        gradient = [np.zeros(layer.shape) for layer in self.model]
        convolutional_gradient = [np.zeros(layer.shape) for layer in self.convolutional_model]
        convolutional_biases_gradient = [np.zeros(layer.shape) for layer in self.convolutional_biases]
        

        dense_layers_count = len(self.model)
        convolutional_layers_count = len(self.convolutional_model)

        output = activations[-1]
        cost = self.cost_function(output, expected_output)

        average_cost = cost.mean()

        old_node_values = None

        for count, layer in enumerate((self.convolutional_model + self.model)[::-1]):  

            index = -(count + 1)     
            output = activations[index]

            if count == (dense_layers_count + convolutional_layers_count):
                break

            elif not count: # Output layer
                num_inputs = self.heights[index - 1]
                height = self.heights[index]

                input_activations = activations[index - 1][:num_inputs]

                cost_derivatives = self.cost_function(output, expected_output, deriv=True)

                activation_derivatives = self.output_layer_activation_function(output, deriv=True)

                node_values = cost_derivatives * activation_derivatives

                weights_derivative = node_values[:, None] * input_activations
                bias_derivative = node_values

                weights = layer[:height, :num_inputs]
                biases = layer[:height, num_inputs]

                w_decay = (2 * weight_decay * weights)
                b_decay = (2 * weight_decay * biases)

                gradient[index][:height, :num_inputs] = weights_derivative - w_decay
                gradient[index][:height, num_inputs] = bias_derivative - b_decay
                old_node_values = node_values

            elif count < dense_layers_count: # Hidden layers

                num_inputs = self.heights[index - 1]
                height = self.heights[index]

                old_height = self.heights[index + 1]
                old_weights = self.model[index + 1][:old_height, :height]

                weights = layer[:height, :num_inputs]
                biases = layer[:height, num_inputs]

                input_activations = activations[index - 1][:num_inputs]

                activation_derivatives = self.hidden_layer_activation_function(output, deriv=True)

                node_values = activation_derivatives * np.dot(old_weights.T, old_node_values)

                if count == dense_layers_count-1: # First feed forward layer
                    input_activations = input_activations[0].flatten()
                    old_node_values = np.dot(weights.T, node_values[:, None])

                else:
                    old_node_values = node_values

                w_decay = (2 * weight_decay * weights)
                b_decay = (2 * weight_decay * biases)

                weights_derivative = node_values[:, None] * input_activations
                bias_derivative = 1 * node_values

                gradient[index][:height, :num_inputs] = weights_derivative - w_decay
                gradient[index][:height, num_inputs] = bias_derivative - b_decay

            else: # Convolutional layers
                depth, scale, pooling_scale = self.convolutional_layers[count - dense_layers_count]

                num_inputs = scale ** 2

                if count == dense_layers_count + convolutional_layers_count - 1: # Last convolutional layer
                    input_activations = activations[index - 1]

                else:
                    input_activations = activations[index - 1][0]

                pooled_outputs, pooling_indices, input_dimensions = output

                output_shape = np.array(pooled_outputs.shape)

                index = -((count - dense_layers_count) + 1)

                if count == dense_layers_count:
                    old_node_values = old_node_values.reshape(output_shape)

                node_values = np.zeros(input_activations.shape)

                # TODO: Add numpy only option

                unpooled_width, unpooled_height = input_dimensions
                iterations_width = np.ceil(unpooled_width / pooling_scale).astype(int)
                iterations_height = np.ceil(unpooled_height / pooling_scale).astype(int)

                for i, kernels in enumerate(layer):
                    kernels = kernels[:depth]
                    for j, (image, kernel, indices) in enumerate(zip(input_activations, kernels, pooling_indices)):

                        unpooled_array = np.zeros((unpooled_width, unpooled_height))
                        for k, pos in enumerate(indices):

                            x = k % iterations_width
                            y = k // iterations_width

                            padded_array = unpooled_array[
                                x * pooling_scale: (x + 1) * pooling_scale, 
                                y * pooling_scale: (y + 1) * pooling_scale
                            ]

                            padded_array[pos[0] % padded_array.shape[0], pos[1] % padded_array.shape[1]] = old_node_values[i][x, y]

                            unpooled_array[
                                x * pooling_scale: (x + 1) * pooling_scale, 
                                y * pooling_scale: (y + 1) * pooling_scale
                            ] = padded_array

                        kernel_decay = 2 * weight_decay * convolutional_gradient[index][i, j]

                        convolutional_gradient[index][i, j] = scipy.signal.convolve2d(image, unpooled_array, "valid") - kernel_decay
                        node_values[j] += scipy.signal.convolve2d(unpooled_array, kernel, "full")

                bias_decay = 2 * weight_decay * convolutional_biases_gradient[index]

                convolutional_biases_gradient[index] = unpooled_array - bias_decay

                old_node_values = node_values

        return gradient, convolutional_gradient, convolutional_biases_gradient, average_cost

    # @jit(forceobj=True, cache=False)
    def eval(self, input, dropout_rate = 0, training=True, numpy_only=False):
        model = self.model
        heights = self.heights
        convolutional_model = self.convolutional_model
        convolutional_layers = self.convolutional_layers
        convolutional_biases = self.convolutional_biases

        length = len(model)
        convolutional_layers_count = len(convolutional_layers)

        input_channels = np.array(input)
        old_height = len(input_channels)

        layer_outputs = [0] * (length + convolutional_layers_count + 1)
        layer_outputs[0] = input_channels

        
        for idx, ((height, scale, pooling_scale), layer, biases) in enumerate(zip(convolutional_layers, convolutional_model, convolutional_biases)):
            layer = np.array(layer)

            shape = (scale, scale)
            pooling_shape = (pooling_scale, pooling_scale)

            # Convolution

            if numpy_only:

                depth, input_width, input_height = input_channels.shape

                input_dimensions = np.array((input_height, input_width))
                result_height, result_width = input_dimensions - shape + 1
                num_samples = result_height * result_width
                
                strides = (input_channels.strides[0],) + input_channels.strides[1:] * 2
                input_samples = np.lib.stride_tricks.as_strided(
                    input_channels,
                    shape=(depth, num_samples, result_height, *shape),
                    strides=strides
                )[:, :result_width, :result_height, :, :]

                output = input_samples * layer[:, :, None, None, :, :]
                output = np.sum(output, axis=(-5, -2, -1)) + biases

            else:
                output = biases.copy()
                for i, kernels in enumerate(layer):
                    for kernel, channel in zip(kernels, input_channels):
                        output[i] += scipy.signal.convolve2d(channel, kernel, "valid")

            # Pooling

            result_dimensions = np.ceil(np.array(output.shape[1:]) / pooling_shape).astype(int)
            result_width, result_height = result_dimensions

            if training:
                output_width, output_height = output.shape[1:]
                # output = np.pad(output, [(0, 0), (0, output_width % pooling_scale), (0, output_height % pooling_scale)])

                pooling_samples = np.lib.stride_tricks.as_strided(
                    output,
                    shape=(height, result_width, result_height, *pooling_shape),
                    strides=(
                        output.strides[0],
                        output.strides[1] * pooling_shape[0],
                        output.strides[2] * pooling_shape[1],
                        *output.strides[1:]
                    )
                )

                pooling_windows = pooling_samples.reshape(height, result_width * result_height, *pooling_shape)
            
            if numpy_only:
                pooled_output = np.max(pooling_windows, axis=(2, 3)).reshape(height, *result_dimensions)

            else:
                pooled_output = skimage.measure.block_reduce(output, (1, pooling_scale, pooling_scale), np.max)
                
            if training:
                pooling_indices = np.zeros((height, pooling_windows.shape[1], 2)).astype(int)

                for i, channel in enumerate(pooling_windows):
                    for j, window in enumerate(channel):
                        index = np.unravel_index(np.argmax(window, axis=None), window.shape)
                        pooling_indices[i, j] = index

                layer_outputs[idx + 1] = [pooled_output, pooling_indices, output.shape[1:]]

            input_channels = np.array(pooled_output)

            old_height = height

        input_activations = input_channels.flatten()

        for idx, (height, layer) in enumerate(zip(heights[1:], model)):

            num_inputs = input_activations.shape[0]

            weights = layer[:height, :num_inputs]
            bias = layer[:height, num_inputs]

            output = np.dot(weights, input_activations) + bias

            # Node Activation
            if idx + 1 == length:
                output_activations = self.output_layer_activation_function(output)

            else:
                output_activations = self.hidden_layer_activation_function(output)

                # Droupout
                if training and dropout_rate:
                    mask = (np.random.rand(*output_activations.shape) > dropout_rate) / ( 1 - dropout_rate)
                    output_activations *= mask

            input_activations = output_activations

            if training or idx + 1 == length:
                layer_outputs[idx + convolutional_layers_count + 1] = output_activations

        return layer_outputs