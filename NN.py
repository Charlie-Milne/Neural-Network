import math
import random
import os

class Node():
    def __init__(self,numOfInputs : int, index : int):
        """
        initializes the weights array and bias to random numbers
        
        :param numOfInputs: the amount of nodes in the previous layer
        :type numOfInputs: int
        :param index: the index of the node within it's layers self.nodes
        :type index: int
        """
        self.index = index
        self.output = 0
        self.weightedSum =0
        self.partialDerivative = 1
        self.previousinputs = [0] * numOfInputs

        self.weights =[]
        for _ in range(numOfInputs):
            self.weights.append(random.random()*2 -1)
        self.bias = random.randint(-1,1)
    

    def calculateOutput(self, inputVector : list):
        """
        calculates the output for the given node
        
        :param inputVector: the outputs for the previous layer in an array
        """
        self.previousinputs = inputVector
        total = 0
        #apply weight to each input
        for index, input in enumerate(inputVector):
            total += input * self.weights[index]
        total += self.bias
        self.weightedSum = total

        #apply activation function (sigmoid)
        total = 1/(1+math.exp(-total))

        self.output = total

    def updateWeights(self,nextLayer, learnRate : float):
        """
        updates each of the weights in for the node using calculus
        to find out how much a change would affect the cost and then
        changes the wieghts based on that.
        
        :param self: the next layer in the NN
        :param learnRate: the value determining how much the weights will be change
        :type learnRate: float
        """
        #using gradient descent

        gradient = 1
        if not isinstance(nextLayer,Layer):
            #last (output) layer
            gradient *= 2*(self.output - nextLayer[self.index])
        else:
            sum = 0
            for node in nextLayer.nodes:#nextLayer is last layer to be updated
                sum += node.weights[self.index] * node.partialDerivative
            gradient *= sum


        gradient *= self.output * (1-self.output) #may be weighted input instead of output

        self.partialDerivative = gradient 

        for index, weight in enumerate(self.weights.copy()):

            gradient = self.partialDerivative
            gradient *= self.previousinputs[index]

            self.weights[index] = weight - learnRate*gradient

        #update bias
        self.bias -= learnRate * self.partialDerivative     

class Layer():
    def __init__(self,numOfNodes : int,numOfInputs : int):
        """
        creates an array of Node objects
        
        :param numOfNodes: the amount of nodes in the layer
        :type numOfNodes: int
        :param numOfInputs: the number of nodes in the previous layer
        :type numOfInputs: int
        """
        self.nodes = []
        for i in range (numOfNodes):
            self.nodes.append(Node(numOfInputs,i))
        self.outputs = [0]*numOfNodes

    def calculateOutputs(self,previousLayersOutputs : list):
        """
        calls the calculate output function for each node in the layer

        :param previousLayersOutputs: an array containing the outputs of the previous layer
        :type previousLayersOutputs: list
        """
        self.outputs=[]
        for node in self.nodes:
            node.calculateOutput(previousLayersOutputs)
            self.outputs.append(node.output)

    def updateWeights(self,nextLayer,learnate : float):
        """
        calls the updateWeights function for each node in the layer
        
        :param nextLayer: the next layer in the NN (the previous one that was updated)
        :param learnate: the value determining how much the weights get changed each update
        :type learnate: float
        """

        for node in self.nodes:
            node.updateWeights(nextLayer, learnate)

class NeuralNetwork():
    def __init__(self,numOfLayers : int,numOfNodes : int,numOfInputs : int,numOfOutputs : int):
        """
        creates all the layers and nodes
        
        :param numOfLayers: the number of hidden layers
        :type numOfLayers: int
        :param numOfNodes: number of nodes in each hidden layer
        :type numOfNodes: int
        :param numOfInputs: number of inputs to the NN
        :type numOfInputs: int
        :param numOfOutputs: the number of outputs
        :type numOfOutputs: int
        """
        self.layers = []
        #add first layer (needs different amount of weights since its only inputs are the inputs)
        self.layers.append(Layer(numOfNodes,numOfInputs))

        for i in range(numOfLayers):
            self.layers.append(Layer(numOfNodes,len(self.layers[i].nodes)))
        
        #add output layer
        self.layers.append(Layer(numOfOutputs,numOfNodes))

    def calculateOutputs(self, inputs : list):
        """
        returns the output of the NN for the given inputs
        
        :param inputs: a list of the inputs
        :type inputs: list
        """

        for layer in self.layers:
            layer.calculateOutputs(inputs)
            inputs = layer.outputs.copy()

        return inputs
    
    def calculateCost(self,inputs : list, expectedOutput : list):
        """
        returns the overall cost of the NN on the given inputs
        
        :param inputs: a list of the inputs
        :type inputs: list
        :param expectedOutput: the list of the expected outputs of each output node
        :type expectedOutput: list
        """
        output = self.calculateOutputs(inputs)
        cost = 0
        for index, output in enumerate(output):
            cost += pow(expectedOutput[index]-output,2)
        return cost
    
    def updateWeights(self,inputs : list, expectedOutputs : list, learnrate : float):
        """
        repeatedly moves back between each layer updating the weights of each node in that layer

        :param inputs: the inputs to update the weights on
        :type inputs: list
        :param expectedOutputs: the expected outputs for the given inputs
        :type expectedOutputs: list
        :param learnrate: the value determining how much the weights get changed each update
        :type learnrate: float
        """
        self.calculateOutputs(inputs)

        self.layers[-1].updateWeights(expectedOutputs, learnrate)
        previousLayer = self.layers[-1]

        #update wieghts for all layers backwards
        for index in range(len(self.layers)-2,-1,-1):
            self.layers[index].updateWeights(previousLayer, learnrate)
            previousLayer = self.layers[index]

    def trainModel(self,trainingSet : list[list[list]], epochs: int, learningRate: int):
        """
        Docstring for trainModel
        
        :param trainingSet: A 3d list which contains list of certain 
        data points input and output lists.
        :type trainingSet: list[list[list]]
        :param epochs: The number of repeats of the training data
        :type epochs: int
        :param learningRate: the learning rate of the neural network
        :type learningRate: int
        """
        for repeat in epochs:
            for data in trainingSet:
                self.updateWeights(data[0],data[1],learningRate)

    def print(self):
        """
        prints out each nodes weights and bias'
        """
        for index,layer in enumerate(self.layers):
            print("------------------------")
            print(f"Layer{index}:")
            for nodeIndex,node in enumerate(layer.nodes):
                print(f"Node{nodeIndex}:")
                print("Wieghts: ", *node.weights ,sep=" | ")
                print(f"Bias: {node.bias}")
    
    def save(self,fileName):
        """
        saves the current NN weights and bias'
        
        :param fileName: the name of the file to store the info in
        """
        #each line will a specific nodes weigths and then the last number will be it's bias
        openFile = open(fileName,"w")
        for layer in self.layers:
            for node in layer.nodes:
                line = ','.join(map(str,node.weights))
                line += f",{node.bias}\n"
                openFile.write(line)
            openFile.write(f"@Layer\n")

        openFile.close() 

    def loadFromFile(self,fileName):
        """
        Loads a NN from a file generated by the save function
        
        :param fileName: the name of the file with the saved NN in
        """
        if not os.path.exists(fileName):
            raise Exception("File does not exist")
        
        openFile = open(fileName,"r")
        line = openFile.readline().rstrip()
        newLayers = []
        nodes = []
        while line != '':
            if line == "@Layer":
                #new layer
                #add previous layer to list
                newLayer = Layer(len(nodes),len(nodes[0].weights))
                newLayer.nodes = nodes
                newLayers.append(newLayer)
                nodes = []
            else:
                values = line.split(",")
                newNode = Node(len(values)-1,len(nodes))
                newNode.weights = list(map(float,values[:-1]))
                newNode.bias = float(values[-1])
                nodes.append(newNode)
            line = openFile.readline().rstrip()
        #finished
        self.layers = newLayers


            

    
class main():
    def __init__(self):
        numOfLayers = 1
        numOfNodes = 1

        inputs = [1,1]
        numOfOutputs = 2
        expectedOutput = [1,1]

        learningRate = 0.2

        nn = NeuralNetwork(numOfLayers,numOfNodes,len(inputs),numOfOutputs)
        nn.print()
        print(nn.calculateCost(inputs,expectedOutput))

        for i in range(20):
            nn.updateWeights(inputs, expectedOutput, learningRate)
            print(nn.calculateCost(inputs,expectedOutput))

        print(nn.calculateCost(inputs,expectedOutput))
        
        nn.print()
if __name__ == "__main__":
    main()