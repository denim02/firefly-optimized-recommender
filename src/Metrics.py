def HitRate(topN,predictions):
    total = 0
    hits = 0
    for pred in predictions:
        user_id = pred[0]
        lo_item_id = pred[1]
        hit = False
        for item_id, _ in topN[int(user_id)]:
            if int(lo_item_id) == int(item_id):
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
        lo_item_id = pred[1]
        actual_rating = pred[2]
        if actual_rating >= ratingCutoff:
            hit = False
            for item_id, _ in topN[int(user_id)]:
                if int(lo_item_id) == int(item_id):
                    hit = True
                    break
            if hit:
                hits += 1
            total += 1
    return hits/total