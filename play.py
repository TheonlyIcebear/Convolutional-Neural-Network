import multiprocessing, threading, requests, hashlib, random, numpy as np, math, copy, gzip, time, json, os
from multiprocessing import Process, Queue
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from utils.model import Model
from numba import jit, cuda
from tkinter import Tk, Button, Label
import networkx as nx
from tqdm import tqdm


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

    def visualize_convolutions(self, layers, orignal):
        root = Tk()
        root.geometry("720x500")

        largest_layer = max([len(layer[0]) for layer in layers])
        center = largest_layer // 2
        
        tkimage = ImageTk.PhotoImage(orignal.resize((self.image_width, self.image_height)))
        label = Label(image=tkimage)
        label.grid(row = 0, column = center)
        label.image = tkimage

        for i, layer in enumerate(layers):
            images = layer[0]
            labels = []
            for j, image in enumerate(images):
                img = Image.fromarray(np.uint8(image.T * 255) , 'L').resize((image.shape[0] * 4, image.shape[1] * 4))

                tkimage = ImageTk.PhotoImage(img)

                label = Label(image=tkimage)
                label.grid(row = i+1, column = j)
                label.image = tkimage # Prevent Garbage Collector

        

        Button(text="Quit", command=root.destroy).grid(row=len(layers) + 2, column=center)

        root.mainloop()

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
                image = Image.open(requests.get(url, stream=True).raw).resize((self.image_width, self.image_height))
            else:

                folder = options[choice - 1]
                
                filename = random.choice(os.listdir(folder))
                image = Image.open(f'{folder}\\{filename}').resize((self.image_width, self.image_height))

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

            convolutional_outputs = model_outputs[1: len(convolutional_model)+1]

            self.visualize_convolutions(convolutional_outputs, image)
                    
            answer = np.argmax(model_outputs[-1])
            print(f"It's a {options[answer]}")
            print(model_outputs[-1], '\n\n')

if __name__ == '__main__':
    Main()
