from FireflyAlgo import FireflyAlgo
from Utility import MovieLens
from surprise import SVD, NormalPredictor, KNNBasic
from AlgoEvaluation import Evaluator
import time

def loadMovieLensData():
    ml = MovieLens()
    data = ml.loadData()
    return ml,data

ml, data= loadMovieLensData()

# SVD Hyperparameter tuning using Firefly algorithm to get smallest RMSE value
# Will tune the reg_all, n_factors, n_epochs, lr_all parameters
##### MODIFY HERE #####
paramBounds = {
    'reg_all': (0.01, 0.5),
    'n_factors': (25, 300),
    'n_epochs': (20, 200),
    'lr_all': (0.001, 0.05)
}
fireflies = 3
epochs = 5
useRMSE = True      # Could set to false and use hit-rate to tune instead
autoStop = True     # Automatically stops when RMSE/Hit-rate stop improving
######################

# Starting timer to measure the time taken to run the algorithm
start = time.time()

# Find optimal hyperparameters using Firefly algorithm
fa = FireflyAlgo(data, paramBounds, numFireflies=fireflies, maxEpochs=epochs, useRMSE=useRMSE, autoStop=autoStop)
tuned_SVD_params = fa.solve()

# Stopping timer and printing elapsed time
end = time.time()
print("Time taken to run the algorithm: {} seconds\n".format(end - start))

# Build evaluation object
eval = Evaluator(data)

print("\nRunning evaluation between tuned SVD model, untuned SVD, and Normal Predictor...\n")

# Build tuned SVD, untuned SVD, random models
svdtuned = SVD(reg_all=float(tuned_SVD_params['reg_all']), n_factors=int(tuned_SVD_params['n_factors']), n_epochs=int(tuned_SVD_params['n_epochs']), lr_all=float(tuned_SVD_params['lr_all']))
knn = KNNBasic()
svd = SVD()
random = NormalPredictor()

# Add models to evaluation object
eval.addModel(svdtuned,'SVDtuned')
eval.addModel(svd,'SVD')
eval.addModel(random,'Random')

# Evaluate object = fit models, build topN lists, run prediction/hitrate based metrics
eval.evaluateModel(True)