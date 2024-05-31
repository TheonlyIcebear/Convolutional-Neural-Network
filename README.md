# Convolutional Neural Network🧠
A neural network I trained using vanilla numpy and scipy

# Information ℹ

An updated version of my previous neural networks:
[Tic Tac Toe Neural Network](https://github.com/TheonlyIcebear/Tic-Tac-Toe-Machine-Learning)

[Image Classifier Neural Network](https://github.com/TheonlyIcebear/Image-Recognition-AI)

I trained it on the same images and got similiar results to the feed forward network, yet the training time took much longer, likely due optimizations I didn't implement.

# Usage ⚙

You can run `main.py` however you'd like, to change the paramaters go to the bottom of the file and change the paramaters going to `Main`

### Arguements
 > Tests:<br> 
 > - How many times trials each thread will face the model against the training bot<br>

 > Momentum Conservation
 > - The percent of the old gradient descent that is added to the current gradient descent

 > Learning Rate:<br>
 > - This changes how much and how quickly the model changes. The higher the faster but it may actually miss it's target's value and fail to actually train effectively.

 > Dimensions:<br>
 > - The amount of layers in the model and the amount of nodes within each layer.

 > Threads:<br>
 > - How many CPU Threads the program will use for training

To test the model run play.py, you can enter your own image link or have it pick a random image to classify
