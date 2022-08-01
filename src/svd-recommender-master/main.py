from FireflyAlgo import FireflyAlgo
from MLutils import MovieLens
from surprise import SVD, NormalPredictor, KNNBasic
from EvaluatorScript import Evaluator

def loadMovieLensData():
    ml = MovieLens()
    data = ml.loadData()
    rankings = ml.getPopularityRanking()
    return ml,data,rankings

ml, data, rankings = loadMovieLensData()

# SVD Hyperparameter tuning using Firefly algorithm to get smallest RMSE value
fa = FireflyAlgo(data, numFireflies=3, maxEpochs=1)
tuned_SVD_params = fa.solve()

# Build evaluation object
evaluator = Evaluator(data,rankings)

print("Running evaluation between tuned SVD model, untuned SVD, KNN, and Normal Predictor...\n")

# Build tuned SVD, untuned SVD, random models
svdtuned = SVD(reg_all=tuned_SVD_params['reg_all'],n_factors=tuned_SVD_params['num_factors'],n_epochs=tuned_SVD_params['num_epochs'],lr_all=tuned_SVD_params['lr_all'])
knn = KNNBasic()
svd = SVD()
random = NormalPredictor()

# Add models to evaluation object
evaluator.addModel(svdtuned,'SVDtuned')
evaluator.addModel(svd,'SVD')
evaluator.addModel(knn,'KNN Basic')
evaluator.addModel(random,'Random')

# Evaluate object = fit models, build topN lists, run prediction/hitrate based metrics
evaluator.evaluateModel(True)

# Build topN list for target user 56
evaluator.sampleUser(ml)
