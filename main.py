import reinforcement
import math
import random
import pickle
import matplotlib.pyplot as plt
from reinforcement import NNfilter

# Compare and edit this code based on the existing SRL framework
# Just shape it so that it's GN&C based and also it can use other types of training data
# Use MATLAB

# Connect to MATLAB PID data
def appendToFile(inputValue):
    with open("input.txt", "a") as inputFile:
        inputFile.write(f'{inputValue}\n')

# Make sure this is actually computing for the Q-learning algorithm
def feedback():
    feedbackVal = int(input("Feedback value: "))
    return feedbackVal

def plotting(data):

    plt.plot(data, marker='o')

    plt.title("Title")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.show()


def main():

    filter = NNfilter()

    # Adjust me later

    desiredPoints = 100
    adjustLearningRate = 0.01

    count = 1

    inputWeights = [1, 1.0]
    outputWeights = 1.0
    currentState = []

    # Load inputFile

    with open("input.txt", "r") as inputFile:
        currentState = [float(line.strip()) for line in inputFile.readlines()]
    
    action = input("'E' for existing, any keys fore new: ")

    if (action == 'E'):
        filter.loadModel('trainingdata.pt')

    feedbackVal = 0
    pointsList = []

    desiredPoints = float(input("insert desired value: "))

    while(1):
        print("currentState: ", currentState)
        
        feedbackVal = feedback()
        
        filter.training(currentState, feedbackVal, adjustLearningRate, inputWeights, outputWeights, desiredPoints)

        filter.saveModel('trainingdata.pt')

        predictedState = filter.predict(currentState, inputWeights)

        with open("input.txt", "w") as outputFile:
            outputFile.write(f"{predictedState[-1]}\n")

        appendToFile(predictedState[-1])

        print(f"predictedState: ", predictedState[0])

        pointsList.append(predictedState[0])

        plotting(pointsList)

if __name__ == "__main__":
    main()