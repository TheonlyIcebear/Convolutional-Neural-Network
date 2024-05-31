from multiprocessing import Process, Queue
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from utils.model import Model
from numba import jit, cuda
import networkx as nx
from tqdm import tqdm
import multiprocessing, threading, requests, hashlib, random, numpy as np, math, copy, gzip, time, json, os

with gzip.open('model-training-data.gz', 'rb') as file:
            file = json.loads(file.read().decode())


class Main:
    def __init__(self):
        with gzip.open('model-training-data.gz', 'rb') as file:
            file = json.loads(file.read().decode())

        self.network = [self.numpify(file[0])] + [self.numpify(file[1])] + [self.numpify(file[2])] + [np.array(file[3])]  + [np.array(file[4])] + file[5:]

        self.image_width = 128
        self.image_height = 128

        # multiprocessing.Process(target=self.visualize_network, args=(self.network[1],)).start()

        self.play()

    def numpify(self, input_array, reverse=False):
        if reverse:
            return [array.tolist() for array in input_array]
        return [np.array(array) for array in input_array]

    def play(self):
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

        options = ['airplane', 'bicycle', 'boat', 'motorbus', 'motorcycle', 'seaplane', 'train', 'truck', 'Custom File']

        while True:
            for idx, option in enumerate(options):
                print(f"{idx + 1}. {option.capitalize()}")

            print('\n')

            print("Enter a integer 1-9")
            choice = int(input(">> "))
            
            if choice == 9:
                print("Enter Image Url: ")
                url = input(">> ")
                image = Image.open(requests.get(url, stream=True).raw)
            else:

                folder = options[choice - 1]
                
                filename = random.choice(os.listdir(folder))
                image = Image.open(f'{folder}\\{filename}').resize((self.image_width, self.image_height)).convert('RGB')

            image.show()

            r, g, b = image.split()

            model_outputs = model.eval(
                input=[
                    np.array(r).T / 255, 
                    np.array(g).T / 255, 
                    np.array(b).T / 255
                ],
                numpy_only=False,
                training=True
            )

            answer = np.argmax(model_outputs[-1])
            print(f"It's a {options[answer]}")
            print(model_outputs[-1], '\n\n')

if __name__ == '__main__':
    Main()
