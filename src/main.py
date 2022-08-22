from Utility import MovieLens
from surprise import SVD, NormalPredictor
from AlgoEvaluation import Evaluator

def loadMovieLensData():
    ml = MovieLens()
    return ml.loadData()

data = loadMovieLensData()

##### MODIFY HERE #####
tuned_SVD_params = {
    'reg_all': 0.002987,
    'n_factors': 93,
    'n_epochs': 87,
    'lr_all': 0.002987
}
#######################

# Build evaluation object
eval = Evaluator(data)

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
