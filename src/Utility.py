'''
Script to pre process MovieLens data.

Creates a class named MovieLens with functions loadData, getGenres, getYears, getMovieName, getMovieId, getPopularity, getUserRatings
'''

from surprise import Dataset, Reader, KNNBaseline
import re
import csv
from collections import defaultdict
from surprise.model_selection import train_test_split,LeaveOneOut

class MovieLens:
    movieID_to_name = {}
    name_to_movieID = {}
    ratingsPath = 'ml-csv/ratings.csv'
    moviesPath = 'ml-csv/movies.csv'

    def loadData(self):
        ratingsDataset = Dataset.load_builtin('ml-100k')
        with open('ml-100k/u.item', newline='', encoding='ISO-8859-1') as datfile:
            for row in datfile:
                row = row.split('|')
                movieId, movieName = int(row[0]), row[1]
                self.movieID_to_name[movieId] = movieName
                self.name_to_movieID[movieName] = movieId
                del movieId, movieName
        
        return ratingsDataset

    def getUserRatings(self,user):
        # All ratings for a given user, list of tuples
        userRatings = []
        hitUser = False
        with open(self.ratingsPath,newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader) # below errors without this
            for row in ratingReader:
                #print(row)
                userID = int(row[0])
                if user == userID:
                    movieID = int(row[1])
                    rating = float(row[2])
                    userRatings.append((movieID,rating))
                    hitUser = True
                if (hitUser and (user != userID)):
                    break
        return userRatings

    def getPopularityRanking(self):
        # Popularity ranks = most rated movies
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath,newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(),key=lambda x:x[1],reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings

    def getMovieName(self,movieID):
        return self.movieID_to_name[movieID]

    def getMovieId(self,movieName):
        return self.name_to_movieID[movieName]

class EvaluationData:
    def __init__(self,data,withSim=False):
        self.trainSet, self.testSet = train_test_split(data, test_size=0.25, random_state=0)

        LOOX = LeaveOneOut(1, random_state=1)
        for xtrain, xtest in LOOX.split(data):
            self.LOOX_trainSet = xtrain
            self.LOOX_testSet = xtest
            del xtrain, xtest
        self.LOOX_antitestSet = self.LOOX_trainSet.build_anti_testset()

        self.full_trainSet = data.build_full_trainset()
        self.full_antitestSet = self.full_trainSet.build_anti_testset()
        if withSim:
            sim_options = {'name': 'cosine', 'user_based': False}
            self.simAlgo = KNNBaseline(sim_options=sim_options)
            self.simAlgo.fit(self.full_trainSet)

