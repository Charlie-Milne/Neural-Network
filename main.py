import pygame
import sys
import NN
import math

pygame.init()

fileName = "test.txt"
pixelSize = 20
collumns, rows = 28, 28
bottomHeight = 100
size = (collumns*pixelSize,rows*pixelSize + bottomHeight)
font = pygame.font.Font("freesansbold.ttf",25)
fontSmall = pygame.font.Font("freesansbold.ttf",16)

openFile = open("mnist.arff","r")

numLayers = 3
nodesPerLayer = 30
numInputs = 784
numOutputs = 10
epochs = 1
neuralNet = NN.NeuralNetwork(numLayers,nodesPerLayer,numInputs,numOutputs)
NNsaveFile = "save.txt"


def trainOnMNIST(savedNN = "",trainIfExists = False):
    try:
        neuralNet.loadFromFile(savedNN)
        if not trainIfExists:
            return
    except:
        print("No saved NN detected")
    file = open("mnist.arff","r")
    learnrate = 0.05
    #have to use readline as file is too large to fit into an array
    reachedData = False
    while not reachedData:
        line = file.readline()
        if line == "@data\n":
            reachedData = True
    
    inputs = []
    outputs = [0] * 10
    imagesTrained = 0
    incorrect = 0
    count = 0
    done = True
    previousLine = ""
    while line != "":#haven't finished file (empty line would be "\n")
        line = file.readline()
        values = line.rstrip().split(",")
        for index,value in enumerate(values):
            if index == 0 and not done:#if previous value continues onto this line
                value = previousLine + value
                done = True
            if value == "":
                done=  True
                continue #if line ended with a comma
            elif index == len(values)-1:
                done = False
                previousLine = value
                continue
            inputs.append(float(value))
            count += 1
            if count >= (collumns*rows):
                break
        if count >= (collumns*rows):
            count = 0
            outputs[int(values[-1])] = 1
            neuralNet.updateWeights(inputs,outputs,learnrate)

            imagesTrained += 1
            if imagesTrained % 100 == 0:
                print(f"Cost for previous image: {neuralNet.calculateCost(inputs,outputs)}")
                print(f"Images trained on: {imagesTrained}")

            if imagesTrained>=55000:
                ##test
                output = neuralNet.calculateOutputs(inputs)
                if output.index(max(output)) != int(values[-1]):
                    incorrect += 1

            #reset ready for next image
            outputs[int(values[-1])] = 0
            inputs = []
            done = True

    print(f"Accuraccy on last {imagesTrained-55000} images: {(1-(incorrect/(imagesTrained-55000)))*100:.2f}%")
    neuralNet.save(NNsaveFile)

def testOnMNIST(testSize = 0):
    file = open("mnist.arff","r")
    #have to use readline as file is too large to fit into an array
    reachedData = False
    while not reachedData:
        line = file.readline()
        if line == "@data\n":
            reachedData = True

    inputs = []
    imagesTrained = 0
    incorrect = 0
    count = 0
    done = True
    previousLine = ""
    while line != "" and imagesTrained < testSize:#haven't finished file (empty line would be "\n")
        line = file.readline()
        values = line.rstrip().split(",")
        for index,value in enumerate(values):
            if index == 0 and not done:#if previous value continues onto this line
                value = previousLine + value
                done = True
            if value == "":
                done=  True
                continue #if line ended with a comma
            elif index == len(values)-1:
                done = False
                previousLine = value
                continue
            inputs.append(float(value))
            count += 1
            if count >= (collumns*rows):
                break
        if count >= (collumns*rows):
            #finished reading inputs
            imagesTrained += 1
            output = neuralNet.calculateOutputs(inputs)
            if output.index(max(output)) != int(values[-1]):
                incorrect += 1
            inputs = []
            count = 0
            done = True

    print(f"Accuraccy on {testSize} images: {(1-(incorrect/(testSize)))*100:.2f}%")


    

class displayImage():
    def __init__(self):
        self.number = "N/A"
        self.prediction = "N/A"

        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(f"Displaying: {fileName.split('.')[0]}")
        self.clock = pygame.time.Clock()

        self.rightArrow = self.renderArrow()
        self.arrowHitbox = pygame.Rect(size[0]-self.rightArrow.get_width(),size[1]//2 - self.rightArrow.get_height()//2,self.rightArrow.get_width(),self.rightArrow.get_height())

        self.pixels = []

    def renderArrow(self):
        #create right arrow
        rightArrow = pygame.image.load("rightArrow.png").convert_alpha()
        rightArrow = pygame.transform.scale(rightArrow,(70,110))
        rightArrow.set_alpha(120)
        return rightArrow
    
    def renderMetaData(self, drawing = False):
        text = f"Number: {self.number}" if not drawing else "Clear"
        numberText = font.render(text,True,(255,255,255),(0,0,0))
        textRect = numberText.get_rect()
        rect = pygame.Rect(0,size[1]-bottomHeight,160,bottomHeight)
        textRect.center = rect.center
        self.numberOutline, self.numberText,self.textRect =rect, numberText, textRect

    def renderTestButton(self,pressed = False):
        #draw test NN button
        if not pressed:
            testText = font.render("Test on AI",True, (255,255,255),(0,0,0))
        else:
            testText = font.render(f"Prediction: {self.prediction}",True, (255,255,255),(0,0,0))
        
        textRect = testText.get_rect()
        testOutline = pygame.Rect(self.numberOutline.width,size[1]-bottomHeight,200,bottomHeight)
        textRect.center = testOutline.center
        self.testOutline, self.testText,self.testRect = testOutline, testText, textRect

    def renderDrawButton(self, drawing = False):
        text = "Draw" if not drawing else "Stop Drawing"
        drawText = font.render(text,True,(255,255,255),(0,0,0))
        textRect = drawText.get_rect()
        left = self.numberOutline.width + self.testOutline.width
        drawOutline = pygame.Rect(left,size[1]-bottomHeight,size[0]-left,bottomHeight)
        textRect.center = drawOutline.center
        self.drawOutline, self.drawText, self.drawRect = drawOutline,drawText,textRect

    def drawInfo(self):
        #drawn number info
        pygame.draw.rect(self.screen,(255,255,255),self.numberOutline,5)
        self.screen.blit(self.numberText,self.textRect)

        #draw test NN button
        pygame.draw.rect(self.screen,(255,255,255),self.testOutline,5)
        self.screen.blit(self.testText,self.testRect)

        #draw "draw" button
        pygame.draw.rect(self.screen,(255,255,255),self.drawOutline,5)
        self.screen.blit(self.drawText,self.drawRect)

    def drawImage(self,file):
        openFile = open(file,"r")
        count = 0
        self.pixels = []
        done = True
        for line in openFile.readlines():
            line = line.rstrip() #remove newline character
            values = line.split(',') # seperates line into an array of numebrs
            for index, value in enumerate(values):
                if index == 0 and not done:#if previous value continues onto this line
                    value = previousLine + value
                    done = True
                if value == "":
                    done=  True
                    continue #if line ended with a comma
                elif index == len(values)-1:
                    done = False
                    previousLine = value
                    continue
                self.pixels.append(float(value))
                count += 1
                if count >= (collumns*rows):
                    break
            print(count)
            if count >= (collumns * rows):
                self.number = values[-1]
                print(self.number)
                break
        
        openFile.close()

    def loadMNIST(self,file,reachedData = False):

        #have to use readline as file is too large to fit into an array
        while not reachedData:
            line = file.readline()
            if line == "@data\n":
                reachedData = True

        #now at actual image data
        done = False
        count = 0
        self.pixels = []
        while not done:
            line = file.readline().rstrip()
            values = line.split(',') # seperates line into an array of numebrs
            for pixel in values:
                self.pixels.append(float(pixel))
                count += 1
                if count >= (collumns * rows):
                    done = True
                    break
            if done:
                self.number = values[-1]

    def drawFromPixels(self,pixels):
        for count,pixel in enumerate(pixels):
            row = count // collumns
            collumn = count - row * collumns
            rect = pygame.Rect(collumn*pixelSize,row*pixelSize,pixelSize,pixelSize)
            colour = int(pixel * 255)
            pygame.draw.rect(self.screen,(colour,colour,colour),rect)

    def testNeuralNet(self,pixels):
        answer = [0]*10
        answer[int(self.number)] = 1

        prediction = neuralNet.calculateOutputs(pixels)

        previousMax = -1
        index = -1
        sum = 0
        for i,value in enumerate(prediction):
            sum += value
            if value > previousMax:
                previousMax = value
                index = i

        self.prediction = index
        self.accuraccy = (previousMax/sum)*100 
        
    def checkHovering(self,mouse):
        hovering = False
        onTestButton = False
        onDrawButton = False
        self.rightArrow.set_alpha(120)
        self.testText.set_alpha(180)
        self.drawText.set_alpha(180)


        if self.arrowHitbox.collidepoint(mouse):
            self.rightArrow.set_alpha(255)
            hovering = True
        elif self.testRect.collidepoint(mouse):
            onTestButton = True
            self.testText.set_alpha(255)
        elif self.drawRect.collidepoint(mouse):
            onDrawButton = True
            self.drawText.set_alpha(255)
        return hovering, onTestButton, onDrawButton

    def drawScreen(self):
        self.screen.fill("black")
        #check every 1 pixels arround mouse 
        pixelRaius = 2
        radius = pygame.Rect(0,0,2*pixelRaius*pixelSize,2*pixelRaius*pixelSize)
        radius.center = pygame.mouse.get_pos()
        #pygame.draw.rect(self.screen,"green",radius)

        mouseButton1 = pygame.mouse.get_pressed()[0]

        if mouseButton1:
            #change pixel vallues
            for count,pixel in enumerate(self.drawingPixels.copy()):
                row = count // collumns
                collumn = count - row * collumns
                rect = pygame.Rect(collumn*pixelSize,row*pixelSize,pixelSize,pixelSize)
                if rect.colliderect(radius):
                    #change colour value
                    dist = math.sqrt(math.pow(rect.centerx-pygame.mouse.get_pos()[0],2)+math.pow(rect.centery-pygame.mouse.get_pos()[1],2))
                    newColour = 1-(dist)/(pixelRaius*pixelSize)
                    self.drawingPixels[count] = max(newColour,pixel)
                    pixel = self.drawingPixels[count]

                colour = int(pixel * 255)
                pygame.draw.rect(self.screen,(colour,colour,colour),rect)
        else:
            self.drawFromPixels(self.drawingPixels)
        
        _,onTest,onDraw = self.checkHovering(pygame.mouse.get_pos())
        if self.numberOutline.collidepoint(pygame.mouse.get_pos()):
            onClear = True
            self.numberText.set_alpha(255)
        else:
            onClear = False
            self.numberText.set_alpha(180)

        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit("See you later")
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if onClear:
                        self.drawingPixels = [0]*754
                        self.renderTestButton(False)
                    elif onTest:
                        try:
                            self.testNeuralNet(self.drawingPixels)
                            self.renderTestButton(True)
                        except Exception as e:
                            print("no applicable AI")
                            print(e)
                    elif onDraw:
                        return False

        self.drawInfo()
        pygame.display.update()

        return True

    def run(self): 
        #self.drawImage(fileName)
        self.loadMNIST(openFile)

        self.renderMetaData()
        self.renderTestButton()
        self.renderDrawButton()

        drawing = False

        while True:
            if drawing:
                stillDrawing = self.drawScreen()
                if stillDrawing:
                    continue
                else:
                    drawing = False
                    self.renderMetaData()
                    self.renderTestButton()
                    self.renderDrawButton()

            onArrow,onTest,onDraw = self.checkHovering(pygame.mouse.get_pos())
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit("See you later")
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if onArrow:
                        self.loadMNIST(openFile,True)
                        self.renderMetaData()
                        self.renderTestButton(False)
                    elif onTest:
                        try:
                            self.testNeuralNet(self.pixels)
                            self.renderTestButton(True)
                        except Exception as e:
                            print("no applicable AI")
                            print(e)
                    elif onDraw:
                        drawing = True
                        self.renderMetaData(True)
                        self.renderDrawButton(True)
                        self.renderTestButton(False)
                        self.drawingPixels = [0]*754
                    
            
            self.screen.fill("black")#erase previous Image
            self.drawFromPixels(self.pixels)
            #make arrow light up when hovering over it.
            
           
            self.screen.blit(self.rightArrow,(size[0]-self.rightArrow.get_width(),size[1]//2 - self.rightArrow.get_height()//2))
            self.drawInfo()
            pygame.display.update()

if __name__ == "__main__":
    for _ in range(epochs):
        trainOnMNIST(NNsaveFile,True)

    #testOnMNIST(6000)
    screen = displayImage()
    screen.run()