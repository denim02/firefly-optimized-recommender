import random as rand
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
import AlgoEvaluation as eval
import numpy as np

class Firefly:
    def __init__(self, dimensions):
        self.position = np.zeros(dimensions)
        self.error = 0.0
        self.intensity = 0.0

class FireflyAlgo:
    def __init__(self, data, params, numFireflies = 40, dimensions = 4, maxEpochs = 1000, seed = 0) -> None:
        self.numFireflies = numFireflies
        self.dimensions = dimensions
        self.maxEpochs = maxEpochs
        self.seed = seed
        self.data = data
        self.evaluator = eval.Evaluator(data)

        # reg_all, n_factors, n_epochs, lr_all
        self.minX = [params['reg_all'][0], params['n_factors'][0], params['n_epochs'][0], params['lr_all'][0]]
        self.maxX = [params['reg_all'][1], params['n_factors'][1], params['n_epochs'][1], params['lr_all'][1]]

        self.trainSet, self.testSet = train_test_split(data, test_size=0.25, random_state=0)

        print("\nInitializing Firefly Algorithm with the following parameters...\n")
        print("Number of fireflies:", numFireflies)
        print("Dimensions:", dimensions)
        print("Max epochs:", maxEpochs)
        print("Seed:", seed)
        print("Min values:", self.minX)
        print("Max values:", self.maxX, "\n")

    def error(self, position):          # Error function checks the inverse of the hitrate of the model
        print("\nTraining model with the following parameters: ")
        print("reg_all:", str(round(position[0], 6)).ljust(10), "| n_factors:", str(int(position[1])).ljust(6), "| n_epochs:", str(int(position[2])).ljust(6), "| lr_all:", str(round(position[3], 6)).ljust(9))
        algo = SVD(reg_all=position[0],n_factors=int(position[1]),n_epochs=int(position[2]),lr_all=position[3])
        
        self.evaluator.addModel(algo, 'SVDtuned')
        hitrate = self.evaluator.getModelHitRate('SVDtuned')
        self.evaluator.clearModels()

        print("Hitrate:", hitrate)

        return 1 / hitrate 
        
    def solve(self):

        B0 = 1.0
        g = 1.0
        a = 0.20

        displayInterval = self.maxEpochs / 5
 
        # Initialize array that will contain fireflies and the result array
        bestError = np.finfo(np.float64).max
        bestPositions = np.zeros(self.dimensions)
        swarm = [None] * self.numFireflies

        print("Initializing fireflies...\n")

        for i in range(0, self.numFireflies):
            # Initialize fireflies and their positions
            swarm[i] = Firefly(self.dimensions)
            for k in range(0, self.dimensions):
                swarm[i].position[k] = rand.uniform(self.minX[k], self.maxX[k])
            
            swarm[i].error = self.error(swarm[i].position)
            swarm[i].intensity = 1 / (swarm[i].error + 1)
            if(swarm[i].error < bestError):
                bestError = swarm[i].error
                bestPositions = swarm[i].position

        print("-------------------------------")
        
        # Main loop
        for epoch in range(0, self.maxEpochs):
            if (epoch % displayInterval == 0):
                print("\n-----------")
                print("Epoch:", str(epoch + 1).ljust(6), "Best hitrate:", 1 / bestError)

            for i in range(0, self.numFireflies):
                for j in range(0, self.numFireflies):
                    if(swarm[i].intensity < swarm[j].intensity):
                        # Move firefly i towards j
                        r = np.linalg.norm(swarm[i].position - swarm[j].position)
                        beta = np.exp(-g * r ** 2) * B0
                        
                        for k in range(0, self.dimensions):
                            swarm[i].position[k] += beta * (swarm[j].position[k] - swarm[i].position[k])
                            swarm[i].position[k] += a * (rand.random() - 0.5)

                            # Computed value fell outside the bounds of allowed values, therefore recompute a random value to assign
                            if (swarm[i].position[k] < self.minX[k] or swarm[i].position[k] > self.maxX[k]):
                                swarm[i].position[k] = rand.uniform(self.minX[k], self.maxX[k])

                swarm[i].error = self.error(swarm[i].position)
                swarm[i].intensity = 1 / (swarm[i].error + 1)

            swarm.sort(key=lambda x: x.error, reverse=False)
            if(swarm[0].error < bestError):
                bestError = swarm[0].error
                for k in range(0, self.dimensions):
                    bestPositions[k] = swarm[0].position[k]

        print("\nBest Hitrate:", 1 / bestError)
        print("Best Parameters  -->  reg_all:", bestPositions[0], "| n_factors:", int(bestPositions[1]), "| n_epochs:", int(bestPositions[2]), "| lr_all:", bestPositions[3],"\n")
        return {'reg_all': bestPositions[0], 'n_factors': int(bestPositions[1]), 'n_epochs': int(bestPositions[2]), 'lr_all': bestPositions[3]}

