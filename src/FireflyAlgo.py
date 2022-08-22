import random as rand
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
import AlgoEvaluation as eval
import numpy as np

class Firefly:
    def __init__(self, dimensions):
        self.position = np.zeros(dimensions)
        self.intensity = 0.0

class FireflyAlgo:
    def __init__(self, data, paramBounds, numFireflies = 40, dimensions = 4, maxEpochs = 1000, seed = 0, useRMSE = True, autoStop = False) -> None:
        self.numFireflies = numFireflies
        self.dimensions = dimensions
        self.maxEpochs = maxEpochs
        self.seed = seed
        self.data = data
        self.useRMSE = useRMSE
        self.autoStop = autoStop
        self.evaluator = eval.Evaluator(data)

        # reg_all, n_factors, n_epochs, lr_all
        self.minX = [paramBounds['reg_all'][0], paramBounds['n_factors'][0], paramBounds['n_epochs'][0], paramBounds['lr_all'][0]]
        self.maxX = [paramBounds['reg_all'][1], paramBounds['n_factors'][1], paramBounds['n_epochs'][1], paramBounds['lr_all'][1]]

        self.trainSet, self.testSet = train_test_split(data, test_size=0.25, random_state=0)

        print("\nExecuting Firefly Algorithm with the following parameters...\n")
        print("Number of fireflies:", numFireflies)
        print("Max epochs:", maxEpochs)
        print("Dimensions:", dimensions)
        print("Seed:", seed)
        print("Tuning using:", "RMSE" if useRMSE else "Hit-rate")
        print("Auto-Stop:", autoStop)
        print("Min values:", self.minX)
        print("Max values:", self.maxX, "\n")

    def intensity(self, position):
        print("\nTraining model with the following parameters: ")
        print("reg_all:", str(round(position[0], 6)).ljust(10), "| n_factors:", str(int(position[1])).ljust(6), "| n_epochs:", str(int(position[2])).ljust(6), "| lr_all:", str(round(position[3], 6)).ljust(9))
        algo = SVD(reg_all=position[0],n_factors=int(position[1]),n_epochs=int(position[2]),lr_all=position[3])

        if(self.useRMSE):
            return 1 / self.rmse(algo)
        else:
            return self.hitrate(algo)

    def rmse(self, algo):          # Error function checks the RMSE
        algo.fit(self.trainSet)
        predictions = algo.test(self.testSet)
        return accuracy.rmse(predictions, verbose=True)
        
    def hitrate(self, algo):
        self.evaluator.addModel(algo, 'SVDtuned')
        hitrate = self.evaluator.getModelHitRate('SVDtuned')
        self.evaluator.clearModels()

        return hitrate 

    def solve(self):

        B0 = 1.0
        g = 1.0
        a = 0.20

        displayInterval = self.maxEpochs / 10
 
        # Initialize array that will contain fireflies and the result array
        bestIntensity = np.finfo(np.float64).min
        bestPositions = np.zeros(self.dimensions)
        swarm = [None] * self.numFireflies

        print("Initializing fireflies...")

        for i in range(0, self.numFireflies):
            # Initialize fireflies and their positions
            swarm[i] = Firefly(self.dimensions)
            for k in range(0, self.dimensions):
                swarm[i].position[k] = rand.uniform(self.minX[k], self.maxX[k])
            
            swarm[i].intensity = self.intensity(swarm[i].position)
            if(swarm[i].intensity > bestIntensity):
                bestIntensity = swarm[i].intensity
                bestPositions = swarm[i].position

        print("\nInitialization finished...")
        
        # The loop will automatically stop if the bestIntensity doesn't improve by more than 0.5% after 5 iterations (if autoStop is set to true)
        iter_count = 0

        # Main loop
        for epoch in range(0, self.maxEpochs):
            if (epoch % displayInterval == 0):
                print("\n-----------")
                print("Epoch:", str(epoch + 1).ljust(6), "Best " + ("RMSE" if self.useRMSE else "Hit-rate") + ": ", (1 / bestIntensity if self.useRMSE else bestIntensity))
                print("-----------")
 
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

                swarm[i].intensity = self.intensity(swarm[i].position)

            swarm.sort(key=lambda x: x.intensity, reverse=True)
            prevBestIntensity = bestIntensity
            if(swarm[0].intensity > bestIntensity):
                bestIntensity = swarm[0].intensity
                for k in range(0, self.dimensions):
                    bestPositions[k] = swarm[0].position[k]
            
            if(self.autoStop):
                if(abs(prevBestIntensity - bestIntensity) < 0.005):
                    iter_count += 1
                if(iter_count == 5):
                    print("\nAutomatically ending calculation after 5 iterations without change --> Epoch:", epoch)
                    print("\n-------------------------------")
                    break
                
        print("\n--------------------------------")
        print("Best " + ("RMSE" if self.useRMSE else "Hit-rate") + ": ", (1 / bestIntensity if self.useRMSE else bestIntensity))
        print("Best Parameters  -->  reg_all:", bestPositions[0], "| n_factors:", int(bestPositions[1]), "| n_epochs:", int(bestPositions[2]), "| lr_all:", bestPositions[3])
        print("--------------------------------\n")
        return {'reg_all': bestPositions[0], 'n_factors': int(bestPositions[1]), 'n_epochs': int(bestPositions[2]), 'lr_all': bestPositions[3]}

