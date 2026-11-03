# Neural Network in Python

<img width="642" height="800" alt="image" src="https://github.com/user-attachments/assets/c4826961-7959-4260-90bb-e537bc08f1a1" />

Within this project the `NN.py` file contains all the code encapsulating the Neural network.
<br></br>
By default the NN framework uses ReLu as the activation function for all nodes in each hidden 
layer and then uses sigmoid for the final output layer.

To change this simply edit line 173 in `NN.py` to:
```python
self.activationFunc = sigmoid
```
This will make all the nodes in the hidden layers use the sigmoid activation function. If you also want to change the output
layers activation function to ReLu you can replace line 181 and 325 with:
```python
self.layers.append(Layer(numOfOutputs,numOfNodes, relu))
activationFunc = relu
```
respectively. The `main` class just shows a small example of how you would create use and train a NN using this framework. 
The `save` and `loadFromFile` methods can be used to save an load you current NN to and from the .txt file with the name provided.
<br>
The `main.py` file is then where I have used this framework along with pygame to create and interactively test a NN built using
the framework from `NN.py` and then trained using 50,000 hand written digits from the MNIST database stored in a csv format.

Using this framework my NN was able to achieve 97% accuracy with 3 epochs.

Below is an example of what the pygame window looked like before and after testing the AI on which number it could see:

<img width="649" height="800" alt="image" src="https://github.com/user-attachments/assets/36a74409-4850-4e79-a978-c662c7a29037" />
<img width="646" height="800" alt="image" src="https://github.com/user-attachments/assets/67d07b32-bf2f-4579-aafc-3b50f84bc041" />

Clicking the arrows will then render new handrawn digits from the dataset and allow you to test your NN on it. You can also use the
draw option to draw your own digits and see what digit the NN thinks you drew.
