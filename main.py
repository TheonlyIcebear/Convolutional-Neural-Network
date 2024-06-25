from multiprocessing import Manager, Process, Queue, Pool
from utils.model import Model
from numba import jit, cuda
from PIL import Image
from tqdm import tqdm
import multiprocessing, threading, hashlib, random, numpy as np, copy, gzip, math, time, json, sys, os

class Main:
    def __init__(self, tests_amount, generation_limit, dimensions, convolutional_dimensions, outputs, hidden_layer_activation_function="relu", output_layer_activation_function="softmax", cost_function="cross_entropy", optimization="adam", learning_rate=0.01, momentum_constant=0.9, adam_constant=0.99, training_percent=0.85, weight_decay=0, dropout_rate=0, cost_limit=0.01, variance=None, annealing_constant=0.0002, threads=4, image_width=128, image_height=128, input_channels=3):
        self.tests = tests_amount
        self.trials = generation_limit
        self.optimization = optimization
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant
        self.adam_constant = adam_constant
        self.wd = weight_decay
        self.dropout_rate = dropout_rate
        self.layers, self.height = dimensions[0]
        self.layers += 1
        self.convolutional_dimensions = convolutional_dimensions
        self.shape = dimensions[1]
        self.threads = threads
        self.cost_limit = cost_limit
        self.variance = variance
        self.annealing_constant = annealing_constant

        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_activation_function
        self.cost_function = cost_function
        
        self.save_queue = Manager().Queue()
        self.queue = Manager().Queue()
        self.children = 0
        self.generations = 0
        self.average_cost = 0
        self.model = []
        self.image_width = image_width
        self.image_height = image_height

        self.input_channels = input_channels

        self.percent = training_percent
        self.outputs = outputs

        # self.choices = np.random.choice(files, size=int(files * self.percent))

        try:
            with gzip.open('model-training-data.gz', 'rb') as file:
                file = json.loads(file.read().decode())

            network = [self.numpify(file[0])] + [self.numpify(file[1])] + [self.numpify(file[2])] + [np.array(file[3])]  + [np.array(file[4])] + file[5:]
        except Exception as e:
            print(e)
            network = self.build()
            open('model-training-data.gz', 'w+').close()

        print(network[3:])

        self.network = network

        threading.Thread(target=self.save).start()
        self.manager(network)
        

    def numpify(self, input_array, reverse=False):
        if reverse:
            return [array.tolist() for array in input_array]
        return [np.array(array) for array in input_array]

    def join_gradients(self, gradients):
        new_gradient = [np.zeros(np.array(layer).shape) for layer in gradients[0]]

        for gradient in gradients:
            for idx, layer in enumerate(gradient):
                new_gradient[idx] += layer

        return new_gradient

    def apply_gradient(self, values, gradient, descent_values):
        new_values = []
        new_gradient = []

        epsilon = 10 ** -8

        learning_rate = self.learning_rate

        if descent_values is None:
            descent_values = [None] * len(values)

        generation = self.generations + 1

        if self.optimization == "adam":

            for idx, (layer, layer_gradient, layer_descent_values) in enumerate(zip(values, gradient, descent_values)):

                if not (layer_descent_values is None):

                    momentum, squared_momentum = layer_descent_values
                    
                else:
                    momentum = 0
                    squared_momentum = 0

                gradient_momentum = (self.momentum_constant * momentum) + (1 - self.momentum_constant) * (layer_gradient)
                squared_gradient_momentum = (self.adam_constant * squared_momentum) + (1 - self.adam_constant) * (layer_gradient ** 2)

                gradient_momentum /= 1 - (self.momentum_constant ** generation)
                squared_gradient_momentum /= (1 - self.adam_constant ** generation)

                new_layer = layer - (gradient_momentum * (learning_rate/(np.sqrt(squared_gradient_momentum) + epsilon)))

                new_values.append(new_layer)
                new_gradient.append([gradient_momentum, squared_gradient_momentum])
                
        elif self.optimization == "RMSProp":
            for idx, (layer, layer_gradient, layer_descent_values) in enumerate(zip(values, gradient, descent_values)):

                if not (layer_descent_values is None):
                    squared_momentum = layer_descent_values
                    
                else:
                    squared_momentum = 0

                squared_gradient_momentum = (self.momentum_constant * squared_momentum) + (1 - self.momentum_constant) * (layer_gradient ** 2)
                squared_gradient_momentum /= 1 - (self.momentum_constant ** generation)

                new_layer = layer - (layer_gradient * (learning_rate/(np.sqrt(squared_gradient_momentum) + epsilon)))

                new_values.append(new_layer)
                new_gradient.append(squared_gradient_momentum)

        elif self.optimization == "momentum":
            for idx, (layer, layer_gradient, layer_descent_values) in enumerate(zip(values, gradient, descent_values)):
                if layer_descent_values is None:
                    layer_descent_values = 0

                change =  ( layer_gradient * learning_rate ) + ( layer_descent_values * self.momentum_constant )

                new_layer = layer - change
                new_values.append(new_layer)

                new_gradient.append(change)

        else:
            for idx, (layer, layer_gradient, layer_descent_values) in enumerate(zip(values, gradient, descent_values)):
                change =  ( layer_gradient * learning_rate )

                new_layer = layer - change
                new_values.append(new_layer)

        return new_values, new_gradient

    def save(self):
        queue = self.save_queue

        while True:
            network = self.network

            try:
                data = json.dumps([self.numpify(network[0], reverse=True), self.numpify(network[1], reverse=True), self.numpify(network[2], reverse=True)] + [network[3].tolist()] + [network[4].tolist()] + network[5:]).encode()
            except Exception as e:
                print(e)
                data = None

            if data:
                with gzip.open('model-training-data.temp', 'wb') as file:
                    file.write(data)
                    file.close()
                
                try:
                    os.remove('model-training-data.gz')
                except Exception as e:
                    print(e)
                    pass

                os.rename('model-training-data.temp', 'model-training-data.gz')

    # Manages threads
    def manager(self, network):
        cost_overtime = []
        gradient_momentum = None
        convolutional_gradient_momentum = None
        convolutional_biases_gradient_momentum = None

        # with open('training-files.json', 'w+') as file:
        #     json.dump(self.choices.tolist(), file)

        convolutional_model, model, convolutional_biases = network[:3]
        pool = Pool(processes=self.threads)

        for _ in range(self.trials):

            network = [convolutional_model, model, convolutional_biases] + network[3:]
            
            self.model = model
            self.network = network
            self.generations += 1 
            
            cost = 0
            gradients = []
            convolutional_gradients = []
            convolutional_biases_gradients = []

            start_time = time.time()

            pool.map(self.worker, range(self.threads))
            
            for _ in tqdm(range(self.threads)):
                _cost, gradient_map, convolutional_gradient_map, convolutional_biases_gradient_map = self.queue.get()

                cost += _cost
                gradients.append(gradient_map)
                convolutional_gradients.append(convolutional_gradient_map)
                convolutional_biases_gradients.append(convolutional_biases_gradient_map)

            end_time = time.time()

            gradient = self.join_gradients(gradients)
            convolutional_gradient = self.join_gradients(convolutional_gradients)
            convolutional_biases_gradient = self.join_gradients(convolutional_biases_gradients)

            self.average_cost = cost / (self.threads)

            os.system('cls')

            print(f"Average Cost: {self.average_cost}, Generations: {self.generations}, Live: {self.children}, Learning Rate: {self.learning_rate}, Delay: {end_time-start_time}")

            cost_overtime.append(self.average_cost)

            json.dump(cost_overtime, open('generations.json', 'w+'), indent=2)

            if self.average_cost < self.cost_limit:
                print(f"Cost Minimum Reached: {self.cost_limit}")
                os.system("PAUSE")

            model, gradient_momentum = self.apply_gradient(model, gradient, gradient_momentum)
            convolutional_model, convolutional_gradient_momentum = self.apply_gradient(convolutional_model, convolutional_gradient, convolutional_gradient_momentum)
            convolutional_biases, convolutional_biases_gradient_momentum = self.apply_gradient(convolutional_biases, convolutional_biases_gradient, convolutional_biases_gradient_momentum)

            del gradient, convolutional_gradient, convolutional_biases_gradient

            self.learning_rate *= 1 - self.annealing_constant

    def worker(self, thread_index):
        convolutional_model, model, convolutional_biases, convolutional_layers, heights, hidden_activation_function, output_activation_function, cost_function = self.network
        model = Model(
            model=model,
            heights=heights,
            convolutional_model=convolutional_model,
            convolutional_layers=convolutional_layers,
            convolutional_biases=convolutional_biases,
            hidden_function=hidden_activation_function, 
            output_function=output_activation_function, 
            cost_function=cost_function
        )
        
        cost = 0
        gradients = []
        convolutional_gradients = []
        convolutional_biases_gradients = []

        options = ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck']

        if self.tests % self.threads: # For when the number of tests is not divisible by the amount of threads
            extra = 1 * (thread_index < (self.tests % self.threads))

        else:
            extra = 0

        for _ in range((self.tests // self.threads) + extra):

            choice = random.randint(0, 7)
            
            folder = options[choice]

            filename = random.choice(os.listdir(folder))
            
            r, g, b = Image.open(f'{folder}\\{filename}').resize((self.image_width, self.image_height)).convert('RGB').split()
            
            model_outputs = model.eval(
                input=[np.array(r).T / 255, np.array(g).T / 255, np.array(b).T / 255], 
                dropout_rate=self.dropout_rate,
                numpy_only=False,
                training=True
            )

            expected_output = np.zeros(8)
            expected_output[choice] = 1

            _gradient, _convolutional_gradient, _convolutional_biases_gradient, _cost = model.gradient(model_outputs, expected_output)

            gradients.append(_gradient)
            convolutional_gradients.append(_convolutional_gradient)
            convolutional_biases_gradients.append(_convolutional_biases_gradient)

            cost += _cost

            del _cost, _gradient, _convolutional_gradient, _convolutional_biases_gradient

        gradient = self.join_gradients(gradients)
        convolutional_gradient = self.join_gradients(convolutional_gradients)
        convolutional_biases_gradient = self.join_gradients(convolutional_biases_gradients)

        cost /= self.tests // self.threads

        self.queue.put([cost, gradient, convolutional_gradient, convolutional_biases_gradient])

    def build(self):
        outputs = self.outputs
        shape = self.shape
        
        convolutional_dimensions = self.convolutional_dimensions

        depth = self.input_channels

        input_dimensions = (self.image_width, self.image_height)
        for idx, (depth, kernel_size, pooling_size) in enumerate(convolutional_dimensions):
            result_dimensions = np.array(input_dimensions) - kernel_size + 1
            input_dimensions = np.ceil(result_dimensions / pooling_size).astype(int)
            depth = depth

        width, height = input_dimensions
        inputs = width * height * depth

        # Generate convolutional layers

        input_channels = self.input_channels

        convolutional_model = [*range(len(convolutional_dimensions))]

        for idx, (channels, kernel_size, pool_size) in enumerate(convolutional_dimensions):
            if not self.variance:
                variance = 2 / (input_channels + channels)

            else:
                variance = self.variance

            print(variance)

            convolutional_model[idx] = np.random.uniform(-variance, variance, (channels, input_channels, kernel_size, kernel_size)).tolist()
            input_channels = channels

        # Generate biases for convolutional layers

        input_channels = self.input_channels

        convolutional_biases = [*range(len(convolutional_dimensions))]

        input_dimensions = (self.image_width, self.image_height)
        for idx, (channels, kernel_size, pooling_size) in enumerate(convolutional_dimensions):
            result_dimensions = np.array(input_dimensions) - kernel_size + 1
            input_dimensions = np.ceil(result_dimensions / pooling_size).astype(int)

            if not self.variance:
                variance = 2 / (input_channels + channels)

            else:
                variance = self.variance

            print(variance)

            convolutional_biases[idx] = np.random.uniform(-variance, variance, (channels, *result_dimensions)).tolist()
            input_channels = channels

        # Generate feed forward layer

        model = [0] * self.layers

        heights = np.full(self.layers, self.height)

        if shape:
            heights[[*shape.keys()]] = [*shape.values()]

        heights = np.append([inputs], heights)
        heights[-1] = outputs

        for idx, inputs in enumerate(heights[:-1]):

            outputs = heights[idx + 1]

            if not self.variance:
                variance = 2 / (inputs + outputs)

            else:
                variance = self.variance

            print(variance)

            model[idx] = np.random.uniform(-variance, variance, (outputs, inputs + 1)).tolist()

        return [self.numpify(convolutional_model), self.numpify(model), self.numpify(convolutional_biases), np.array(convolutional_dimensions), np.array(heights), self.hidden_layer_activation_function, self.output_layer_activation_function, self.cost_function]

if __name__ == "__main__":
    
    try:
        Main(
            tests_amount = 24, # The length of the tests,
            generation_limit = 1000000, # The amount of generations the model will be trained through
            training_percent = 0.85, # Percent of data that will be used for training,
            cost_limit = 0.0, # if the cost goes below this number the model will stop training
            
            optimization = "momentum", # momentum, adam, RMSProp, vanilla 
            momentum_constant = 0.99, # What percent of the previous changes that are added to each weight in our gradient descent
            adam_constant = 0.999,
            learning_rate = 0.0001, # How fast the model learns, if too low the model will train very slow and if too high it won't train
            weight_decay = 0.0,
            dropout_rate = 0.0,
            
            variance = 0.15, # If none it will use Xavier initialization to optimize
            annealing_constant = 0.00001,

            hidden_layer_activation_function = "relu", # crelu, softmax, sigmoid, tanh, relu
            output_layer_activation_function = "softmax", # crelu, softmax, sigmoid, tanh, relu
            cost_function = "cross_entropy", # smooth_l1_loss, mse, cross_entropy

            dimensions = [
                [3, 156], 
                {
                    0: 156,
                    1: 64,
                    2: 27,
                }
            ],  # The length and height of the model

            convolutional_dimensions=[
                [3, 3, 2], # how many kernels and the kernel sizes [Kernels, Size, Pooling_Size]
                [6, 3, 2],
                [9, 3, 2]
            ],

            threads = 24,  # How many concurrent threads to be used
            outputs = 8 # There are eight outputs
        )
    except KeyboardInterrupt:
        print("Ctrl + C pressed.\nSafely exiting threads...")
        sys.exit()