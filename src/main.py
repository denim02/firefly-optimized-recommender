from FireflyAlgo import FireflyAlgo
from Utility import MovieLens
from surprise import SVD, NormalPredictor, KNNBasic
from AlgoEvaluation import Evaluator

def loadMovieLensData():
    ml = MovieLens()
    data = ml.loadData()
    rankings = ml.getPopularityRanking()
    return ml,data,rankings

ml, data, rankings = loadMovieLensData()

# SVD Hyperparameter tuning using Firefly algorithm to get smallest RMSE value
# Will tune the reg_all, n_factors, n_epochs, lr_all parameters
##### MODIFY HERE #####
# params = {
#     'reg_all': (0.01, 0.5),
#     'n_factors': (25, 100),
#     'n_epochs': (20, 100),
#     'lr_all': (0.001, 0.05)
# }
# fireflies = 5
# epochs = 10

# # Find optimal hyperparameters using Firefly algorithm
# fa = FireflyAlgo(data, params, numFireflies=fireflies, maxEpochs=epochs)
# tuned_SVD_params = fa.solve()

tuned_SVD_params = {
    'reg_all': 0.002987,
    'n_factors': 93,
    'n_epochs': 87,
    'lr_all': 0.002987
}

# Build evaluation object
eval = Evaluator(data,rankings)

print("\nRunning evaluation between tuned SVD model, untuned SVD, and Normal Predictor...\n")

# Build tuned SVD, untuned SVD, random models
svdtuned = SVD(reg_all=float(tuned_SVD_params['reg_all']), n_factors=int(tuned_SVD_params['n_factors']), n_epochs=int(tuned_SVD_params['n_epochs']), lr_all=float(tuned_SVD_params['lr_all']))
svd = SVD()
random = NormalPredictor()

# Add models to evaluation object
eval.addModel(svdtuned,'SVDtuned')
eval.addModel(svd,'SVD')
eval.addModel(random,'Random')

# Evaluate object = fit models, build topN lists, run prediction/hitrate based metrics
eval.evaluateModel(True)
