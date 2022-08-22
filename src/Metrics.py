import numpy as np
from collections import defaultdict
import itertools

def HitRate(topN,predictions):
    total = 0
    hits = 0
    for pred in predictions:
        user_id = pred[0]
        item_id = pred[1]
        hit = False
        for iid, predictRating in topN[int(user_id)]:
            if int(item_id) == int(iid):
                hit = True
                break
        if hit:
            hits += 1
        total += 1
    return hits/total

def CumulativeHitRate(topN,predictions,ratingCutoff = 0):
    total = 0
    hits = 0
    for pred in predictions:
        user_id = pred[0]
        item_id = pred[1]
        actual_rating = pred[2]
        if actual_rating >= ratingCutoff:
            hit = False
            for iid, predictRating in topN[int(user_id)]:
                if int(item_id) == int(iid):
                    hit = True
                    break
            if hit:
                hits += 1
            total += 1
    return hits/total