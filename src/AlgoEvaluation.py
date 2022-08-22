from collections import defaultdict
import Metrics
from Utility import EvaluationData
from surprise import accuracy
import time

class Evaluator():
    def __init__(self,data):
        self.trainSet,self.testSet,self.LOOX_trainSet,self.LOOX_testSet,self.LOOX_antitestSet, \
            self.full_trainSet,self.full_antitestSet,self.simAlgo = self.processData(data)
        self.models = {}
        self.metrics = {}

    def processData(self,data):
        print('preparing data...')
        eval = EvaluationData(data,True)
        return eval.trainSet,eval.testSet,eval.LOOX_trainSet,eval.LOOX_testSet,eval.LOOX_antitestSet, \
            eval.full_trainSet,eval.full_antitestSet,eval.simAlgo

    def addModel(self,model,name):
        self.models[name] = model

    def evaluateModel(self,doTopN=False):
        for name in self.models:
            t = time.time()
            print('Evaluating',name)
            self.models[name].fit(self.trainSet)
            predictions = self.models[name].test(self.testSet)

            metrics = {}
            metrics['MAE'] = accuracy.mae(predictions)
            metrics['RMSE'] = accuracy.rmse(predictions)

            if doTopN:
                self.models[name].fit(self.LOOX_trainSet)
                LOOX_predictions = self.models[name].test(self.LOOX_testSet)
                LOOXfull_predictions = self.models[name].test(self.LOOX_antitestSet)

                self.models[name].fit(self.full_trainSet)

                LOOX_topN = self.getTopN(LOOXfull_predictions)

                metrics['HR'] = Metrics.HitRate(LOOX_topN, LOOX_predictions)
                metrics['cHR'] = Metrics.CumulativeHitRate(LOOX_topN, LOOX_predictions, 3.0)

            print(metrics)
            print('Time:',time.time() - t)
            print('-----------')
            self.metrics[name] = metrics

    def getTopN(self,predictions, n=10, minRating=3.0):
        topN = defaultdict(list)
        for pred in predictions:
            user, item, predictRating = pred[0], pred[1], pred[3]
            if predictRating >= minRating:
                topN[int(user)].append((int(item), predictRating))

        for user, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(user)] = ratings[:n]
        return topN