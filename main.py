from multiprocessing import Manager, Process, Queue, Pool
from multiprocessing.managers import BaseManager
from utils.model import Model
from PIL import Image
from tqdm import tqdm
import multiprocessing, albumentations as A, threading, hashlib, random, awkward as ak, numpy as np, copy, gzip, math, time, json, sys, os

class Main:
    def __init__(self, batch_size, epoch_limit, dimensions, convolutional_dimensions, hidden_layer_activation_function="relu", output_layer_activation_function="softmax", cost_function="cross_entropy", optimization="adam", save_file="model-training-data.json", learning_rate=0.01, momentum_constant=0.9, adam_constant=0.99, training_percent=0.85, weight_decay=0, dropout_rate=0, cost_limit=0.01, variance=None, annealing_constant=0.0002, threads=4, image_width=480, image_height=270, input_channels=3, outputs=8):
        self.batch_size = batch_size
        self.epoch_limit = epoch_limit
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
        self.save_file = save_file
        
        self.save_queue = Manager().Queue()
        self.queue = Manager().Queue()
        self.generations = 0
        self.average_cost = 0
        self.image_width = image_width
        self.image_height = image_height

        self.input_channels = input_channels
        self.outputs = outputs
        # self.outputs = self.objects * 5

        self.options = ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck']

        self.percent = training_percent

    def numpify(self, input_array, reverse=False):
        if reverse:
            return [array.tolist() for array in input_array]
        return [np.array(array) for array in input_array]

    def join_gradients(self, gradients):
        new_gradient = [np.zeros(np.array(layer).shape) for layer in gradients[0]]

        for idx in tqdm(list(range(len(new_gradient)))):
            new_gradient[idx] = np.sum([gradient[idx] for gradient in gradients], axis=0)

        return new_gradient

    def apply_gradient(self, values, gradient, descent_values):
        new_values = [None] * len(values)
        new_gradient = [None] * len(values)

        epsilon = 10 ** -8

        learning_rate = self.learning_rate

        # if descent_values is None:
        #     descent_values = [None] * len(values)

        generation = self.generations + 1

        if self.optimization == "adam":

            if not isinstance(values, ak.Array):
                values = ak.Array(values)
                gradient = ak.Array(gradient)

            gradient = gradient / self.batch_size # late scaling

            if isinstance(descent_values, ak.Array):
                momentum, squared_momentum = descent_values

                squared_momentum = ak.Array(squared_momentum)
                momentum = ak.Array(momentum)

            else:
                squared_momentum = 0
                momentum = 0

            gradient_momentum = (self.momentum_constant * momentum) + (1 - self.momentum_constant) * gradient
            squared_gradient_momentum = (self.adam_constant * squared_momentum) + (1 - self.adam_constant) * (gradient ** 2)

            gradient_momentum = gradient_momentum / (1 - self.momentum_constant ** generation)
            squared_gradient_momentum = squared_gradient_momentum / (1 - self.adam_constant ** generation)

            new_values = values - (gradient_momentum * (learning_rate/(np.sqrt(squared_gradient_momentum) + epsilon)))
            new_gradient = [momentum, squared_momentum]
                
        elif self.optimization == "RMSProp":

            if not isinstance(values, ak.Array):
                values = ak.Array(values)
                gradient = ak.Array(gradient)

            gradient = gradient / self.batch_size # late scaling

            if isinstance(descent_values, ak.Array):
                squared_momentum = ak.Array(descent_values)

            else:
                squared_momentum = 0

            squared_gradient_momentum = (self.momentum_constant * squared_momentum) + (1 - self.momentum_constant) * (gradient ** 2)
            squared_gradient_momentum = squared_gradient_momentum / (1 - self.momentum_constant ** generation)

            new_values = values - (gradient * (learning_rate/(np.sqrt(squared_gradient_momentum) + epsilon)))
            new_gradient = squared_gradient_momentum

        elif self.optimization == "momentum":
            if not isinstance(values, ak.Array):
                values = ak.Array(values)
                gradient = ak.Array(gradient)

            gradient = gradient / self.batch_size # late scaling

            if isinstance(descent_values, ak.Array):
                momentum = ak.Array(descent_values)

            else:
                momentum = 0

            change = ( gradient * learning_rate ) + ( momentum * self.momentum_constant )

            new_values = values - change
            new_gradient = change

        else:
            if not isinstance(values, ak.Array):
                values = ak.Array(values)
                gradient = ak.Array(gradient)

            gradient = gradient / self.batch_size # late scaling
            new_values = values - gradient

        return self.numpify(new_values), self.numpify(new_gradient)

    def load(self, save_file=None):
        if not save_file:
            save_file = self.save_file

        try:
            print("[LOG] Attempting to load saved file.")
            file = json.load(open(save_file, 'rb'))
            print("[LOG] Successfully loaded file")

            network = [self.numpify(file[0])] + [self.numpify(file[1])] + [self.numpify(file[2])] + [np.array(file[3])]  + [np.array(file[4])] + file[5:]
        except (json.JSONDecodeError, OSError, PermissionError, Exception):
            if os.path.exists(save_file):
                print("[LOG] Malformed data inside file, or file is being used by another process")

            else:
                print("[LOG] File not found")

            print("[LOG] Generating new build file...")

            network = self.build()
            open(save_file, 'w+').close()

        print("[LOG] Starting training")

        print(network[3:])
        
        self.network = network

    def save(self):
        try:
            with open(self.save_file, 'wb') as file:
                file.write(json.dumps([self.numpify(self.network[0], reverse=True), self.numpify(self.network[1], reverse=True), self.numpify(self.network[2], reverse=True)] + [self.network[3].tolist()] + [self.network[4].tolist()] + self.network[5:]).encode())
                file.close()
        except Exception as e:
            print(e)

    # Manages threads
    def train(self):
        cost_overtime = []
        gradient_momentum = None
        convolutional_gradient_momentum = None
        convolutional_biases_gradient_momentum = None

        cost_skipped = False
        epoch_skipped = False

        dataset_size = np.array([len(os.listdir(folder)) for folder in self.options]).sum()

        convolutional_model, model, convolutional_biases, convolutional_layers, heights, hidden_activation_function, output_activation_function, cost_function = self.network
        

        # BaseManager.register('Model', Model)
        # manager = BaseManager()
        # manager.start()


        while True:
            
            self.generations += 1 

            
            epoch = (self.generations * self.batch_size) / (dataset_size * self.percent)
            
            cost = 0
            gradients = [0] * self.batch_size
            convolutional_gradients = [0] * self.batch_size
            convolutional_biases_gradients = [0] * self.batch_size

            # instance = manager.Model(
            #     model=model,
            #     heights=heights,
            #     convolutional_model=convolutional_model,
            #     convolutional_layers=convolutional_layers,
            #     convolutional_biases=convolutional_biases,
            #     hidden_function=hidden_activation_function, 
            #     output_function=output_activation_function, 
            #     cost_function=cost_function
            # )

            # for idx in range(self.threads):
            #     multiprocessing.Process(target=self.worker, args=(idx,)).start()

            start_time = time.time()

            pool = Pool(processes=self.threads)
            threading.Thread(target=pool.map, args=(self.worker, range(self.threads))).start()

            for idx in tqdm(range(self.batch_size)):
                _cost, _gradient, _convolutional_gradient, _convolutional_biases_gradient = self.queue.get()

                cost += _cost

                gradients[idx] = _gradient
                convolutional_gradients[idx] = _convolutional_gradient
                convolutional_biases_gradients[idx] = _convolutional_biases_gradient

                del _cost, _gradient, _convolutional_gradient, _convolutional_biases_gradient
            
            pool.close()
            pool.join()

            gradient = self.join_gradients(gradients)
            convolutional_gradient = self.join_gradients(convolutional_gradients)
            convolutional_biases_gradient = self.join_gradients(convolutional_biases_gradients)
            
            if not self.generations % 10:
                threading.Thread(target=self.save).start()

            end_time = time.time()

            self.average_cost = cost / (self.batch_size)

            if (epoch + 1) != self.epoch_limit:
                os.system('cls')

            elif epoch >= self.epoch_limit and not epoch_skipped:
                print(f"Epoch limit Reached: {self.epoch_limit}")
                print("Press enter to continue training...")
                os.system("PAUSE")
                epoch_skipped = True

            print(f"Average Cost: {self.average_cost}, Generations: {self.generations}, Epoch: {epoch}, Learning Rate: {self.learning_rate}, Delay: {end_time-start_time}")

            cost_overtime.append(self.average_cost)

            json.dump(cost_overtime, open('generations.json', 'w+'), indent=2)

            if self.average_cost < self.cost_limit and not cost_skipped:
                print(f"Cost Minimum Reached: {self.cost_limit}")
                print("Press enter to continue training...")
                os.system("PAUSE")
                cost_skipped = True

            model, gradient_momentum = self.apply_gradient(model, gradient, gradient_momentum)
            convolutional_model, convolutional_gradient_momentum = self.apply_gradient(convolutional_model, convolutional_gradient, convolutional_gradient_momentum)
            convolutional_biases, convolutional_biases_gradient_momentum = self.apply_gradient(convolutional_biases, convolutional_biases_gradient, convolutional_biases_gradient_momentum)

            del gradient, convolutional_gradient, convolutional_biases_gradient

            self.learning_rate *= 1 - self.annealing_constant
            self.network[:3] = [convolutional_model, model, convolutional_biases]

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

        del convolutional_model, convolutional_biases, convolutional_layers, heights, hidden_activation_function, output_activation_function, cost_function, self.network

        if self.batch_size % self.threads: # For when the number of tests is not divisible by the amount of threads
            extra = 1 * (thread_index < (self.batch_size % self.threads))

        else:
            extra = 0

        tests = (self.batch_size // self.threads) + extra # Divides number of tests as equally as possible

        for _ in range(tests):
            choice = random.randint(0, 7)
            
            folder = self.options[choice]

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

            gradient, convolutional_gradient, convolutional_biases_gradient, cost = model.gradient(model_outputs, expected_output)

            while True:
                try:
                    self.queue.put([cost, gradient, convolutional_gradient, convolutional_biases_gradient])
                    break
                except (MemoryError, OSError):
                    continue

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

        print(depth, width, height)

        # Generate convolutional layers

        input_channels = self.input_channels
        input_dimensions = (self.image_width, self.image_height)

        convolutional_model = [*range(len(convolutional_dimensions))]
        convolutional_biases = [*range(len(convolutional_dimensions))]

        for idx, (channels, kernel_size, pool_size) in enumerate(tqdm(convolutional_dimensions)):
            unpooled_dimensions = np.array(input_dimensions) - kernel_size + 1
            result_dimensions = np.ceil(unpooled_dimensions / pool_size).astype(int)

            variance = np.sqrt(2 / (input_channels * kernel_size ** 2))

            convolutional_model[idx] = np.random.normal(0, variance, (channels, input_channels, kernel_size, kernel_size)).tolist()
            convolutional_biases[idx] = np.zeros((channels, *unpooled_dimensions)).tolist()
            input_channels = channels
            input_dimensions = result_dimensions

        # Generate feed forward layer

        model = [0] * self.layers

        heights = np.full(self.layers, self.height)

        if shape:
            heights[[*shape.keys()]] = [*shape.values()]

        heights = np.append([inputs], heights)
        heights[-1] = outputs

        for idx, inputs in enumerate(tqdm(heights[:-1])):

            outputs = heights[idx + 1]

            if not self.variance:
                variance = np.sqrt(2 / (inputs))
            else:
                variance = self.variance

            data = np.random.normal(0, variance, (outputs, inputs + 1))
            data[:, -1] = 0

            model[idx] = data.tolist()

        return [self.numpify(convolutional_model), self.numpify(model), self.numpify(convolutional_biases), np.array(convolutional_dimensions), np.array(heights), self.hidden_layer_activation_function, self.output_layer_activation_function, self.cost_function]

if __name__ == "__main__":

    
    try:
        network = Main(
            batch_size = 32, # The batch size,
            epoch_limit = 1000000000000, # The amount of epochs the model will be trained through
            training_percent = 0.9, # Percent of data that will be used for training,
            cost_limit = 0.0, # if the cost goes below this number the model will stop training
            
            optimization = "adam", # momentum, adam, RMSProp, vanilla 
            momentum_constant = 0.9, # What percent of the previous changes that are added to each weight in our gradient descent
            adam_constant = 0.99,
            learning_rate = 0.0001, # How fast the model learns, if too low the model will train very slow and if too high it won't train
            weight_decay = 0.0,
            dropout_rate = 0.0,
            
            variance = None, # If none it will use He initialization to optimize
            annealing_constant = 0.0,

            hidden_layer_activation_function = "relu", # crelu, softmax, sigmoid, tanh, relu
            output_layer_activation_function = "softmax", # crelu, softmax, sigmoid, tanh, relu
            cost_function = "cross_entropy", # smooth_l1_loss, mse, cross_entropy, weighted_loss

            save_file = "model-training-data.json", # Save file location

            dimensions = [
                [4, 256], 
                {
                    0: 256,
                    1: 128,
                    1: 64,
                    2: 32,
                }
            ],  # The length and height of the model

            convolutional_dimensions=[
                [8, 3, 2], # how many kernels and the kernel sizes [Kernels, Size, Pooling_Size]
                [16, 3, 2],
                [32, 3, 2]
            ],

            image_width = 128,
            image_height = 128,

            outputs = 8,
            threads = 16,  # How many concurrent threads to be used
        )

        network.load()
        network.train()

    except (KeyboardInterrupt, Exception) as e:
        print(e)
        print("Ctrl + C pressed.\nSafely exiting threads...")
        network.save()
        sys.exit()
