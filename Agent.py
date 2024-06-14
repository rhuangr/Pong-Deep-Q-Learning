import random
import numpy as np
import copy
import os
from collections import deque

ACTIONS = ["up", "down"]


#SUPPOSE REWARD OF -1 WHENEVER YOU MISS A BALL, THEN AGENT MIGHT JUST PICK A SPOT FOR WHICH HE HAS A CHANCE OF HITTING THE BALL FOR EXAMPLE IN A CORDER
#TO AVOID NEGATIVE REWARDS IN GENERAL

# this reinforcement learning neural network took SO MUCH MORE TIME than the normal neural network.
# failed to understand bellman equation, was confused about how to substract the target argmax when it is a scalar.
# realized how important, rewards epsilon and learning rate was.
# in the end, I had to guide the agent with small rewards upon following the ball, orelse i got anxious about not converging.

# interesting bug: at first, I gave a reward for the agent hitting the ball. But due to a bug where the ball could
# get stuck behind the pong and bounce off the terminal state wall without ending the game, this would result in the agent
# moving to to clip the ball and bounce it 10 times, rather than consistenly sending the ball back.

class NN:
    
    def __init__(self, layerSizes: list[int]) -> None:
        self.inputSize = layerSizes[0]
        self.hiddenLayerSize = layerSizes[1]
        self.outputSize = layerSizes[2]
        

        self.weights1 = np.random.randn(self.hiddenLayerSize, self.inputSize)
        self.weights2 = np.random.randn(self.outputSize, self.hiddenLayerSize)
        
        self.targetWeights1 = np.zeros_like(self.weights1)
        self.targetWeights2 = np.zeros_like(self.weights2)
        
        self.biases1 = np.random.randn(self.hiddenLayerSize, 1)
        self.biases2 = np.random.randn(self.outputSize, 1)
        
        self.targetBiases1 = np.zeros_like(self.biases1)
        self.targetBiases2 = np.zeros_like(self.biases2)
        
        self.epsilon = 0.8
        self.gamma = 0.95
        self.learningRate = 0.0001
        self.maxReplayMemory = 10_000
        self.batchSize = 175
        self.memory = deque(maxlen=self.maxReplayMemory)
        
        self.tag = None
        self.bounces = 0
        self.goals = 0
        self.i = 0
        self.total = 0
        self.neg = 0

    def getAction(self, input):
        a, z = self.forwardPass(input)
        epsilonCheck = np.random.random()
        action = np.argmax(a[-1])
        if epsilonCheck < self.epsilon:
            action = random.randint(0,1)
        return action
    
    def storeMemory(self, state):
        self.memory.append(state)
        
    def forwardPass(self, input ):
        activationsList = []
        zList = []
        activationsList.append(input)
        hiddenZ = np.dot(self.weights1, np.reshape(input, (self.inputSize, 1))) + self.biases1
        zList.append(hiddenZ)
        activationsList.append(self.sigmoid(hiddenZ))
        outputZ = np.dot(self.weights2, activationsList[1]) + self.biases2
        zList.append(outputZ)
        activationsList.append((outputZ))

        return activationsList, zList
    
    def targetForwardPass(self, input):
        a1 =  self.sigmoid(np.dot(self.targetWeights1, np.reshape(input, (self.inputSize, 1))) + self.targetBiases1)
        a2 = self.sigmoid(np.dot(self.targetWeights2, a1)+ self.targetBiases2) 
        return np.max(a2), np.argmax(a2)
        
    # called on a memory batch, remember value changed from the time when you first computed it
    def backProp(self, stateTuple):
        state, action, reward, nextState, done = stateTuple
        a,z = self.forwardPass(state)
        
        target, targetIndex = self.targetForwardPass(nextState)
        target = reward if done else reward + self.gamma*target
        
        targetVertex = copy.deepcopy(a[-1])
        targetVertex[action] = target
        
        delta = a[-1] - targetVertex
        
        # prevActionV = a[-1][action]

        self.biases2 = self.biases2 - self.learningRate*delta
        self.weights2 = self.weights2 - self.learningRate*(delta * a[-2].T)
        delta = np.dot(self.weights2.T, delta) *  self.sigmoidDerivative(z[-2])
        self.biases1 = self.biases1 -  self.learningRate* delta
        self.weights1 = self.weights1- self.learningRate * np.dot(delta, np.reshape(a[0], (1, self.inputSize)))
        
        # a,z = self.forwardPass(state)
        # if reward != 0:
        #     if reward > 0:
        #         if a[-1][action] - prevActionV < 0:
        #             self.neg += 1
        #         print(f"target: {targetVertex} predicted: {a[-1]}")
        #         print(f"action: {action} reward: {reward} targetAction: {targetIndex}")
        #         print(f"action: {action} changed by: {a[-1][action] - prevActionV}")
                
        #         self.total += 1
        #         print(self.neg/self.total)
        #         print()
        
    def getBatch(self):
        return random.sample(self.memory, self.batchSize)
    
    def updateBatch(self):
        if len(self.memory) < 200:
            return
        batch = self.getBatch()
        for memoryTuple in batch:
            self.backProp(memoryTuple) 
        self.i += 1
        if self.i % 10 == 0:
            self.i = 0
            self.targetWeights1 = copy.deepcopy(self.weights1)
            self.targetWeights2 = copy.deepcopy(self.weights2)
            self.targetBiases1 = copy.deepcopy(self.biases1)
            self.targetBiases2 = copy.deepcopy(self.biases2)
        
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
    
    def relu(self, z):
        for i in range(len(z)):
            z[i] = max(0,z[i])
        return z
     
    def costDerivative(self, predicted, target):
        return (predicted - target)
    
    def sigmoidDerivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def reluDerivative( self, z):
        for i in range(len(z)):
            z[i] = 1 if z[i] >= 0 else 0
        return z
    
    def save_weights(self, file_path):
        np.savez(file_path,
        weights1=self.weights1,
        biases1=self.biases1,
        weights2=self.weights2,
        biases2=self.biases2,
        targetWeights1=self.targetWeights1,
        targetWeights2=self.targetWeights2,
        targetBiases1=self.targetBiases1,
        targetBiases2=self.targetBiases2,
        goals=self.goals,
        epsilon =self.epsilon
        )

    def load_weights(self, file_path):
        if os.path.exists(file_path):
            print(self.weights1)
            data = np.load(file_path)
            self.weights1 = data['weights1']
            self.biases1 = data['biases1']
            self.weights2 = data['weights2']
            self.biases2 = data['biases2']
            self.targetWeights1 = data['targetWeights1']
            self.targetWeights2 = data['targetWeights2']
            self.targetBiases1 = data['targetBiases1']
            self.targetBiases2 = data['targetBiases2']
            self.goals=data['goals']
            self.epsilon=data['epsilon']

            print(self.epsilon)
            