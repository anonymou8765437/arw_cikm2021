import numpy as np
from topology.properties import *
#from fdmf.cython.fdmf_cython import create_negative_mask

def partial_ranking_mask(train_rating_matrix, test_rating_matrix, n_negatives=50):

    mask = test_rating_matrix > 0
    not_rated = (train_rating_matrix + test_rating_matrix) == 0
    items = np.arange(train_rating_matrix.shape[1])
    n_test_rated = np.sum((test_rating_matrix > 0).astype(np.int32), axis=1)
    n_not_rated = np.sum((not_rated > 0).astype(np.int32), axis=1)#
    pick_probs = not_rated.astype(
        np.int32) / np.sum(not_rated.astype(np.int32), axis=1)[:, None]
    for user in range(train_rating_matrix.shape[0]):
        negatives = np.random.choice(
            items, size=min(n_not_rated[user], n_test_rated[user] * n_negatives),  replace=False, p=pick_probs[user])
        mask[user, negatives] = 1
    return mask


def partial_ranking_mask_sparse(train_rating_matrix, test_rating_matrix, n_negatives=50):
    from scipy import sparse 
    mask = test_rating_matrix > 0
    mask = sparse.csrmatrix(test_rating_matrix.shape)
    
    mask[test_rating_matrix.nonzero()] = 1
    
    not_rated = (train_rating_matrix + test_rating_matrix) == 0
    items = np.arange(train_rating_matrix.shape[1])
    n_test_rated = np.sum((test_rating_matrix > 0).astype(np.int32), axis=1)
    n_not_rated = np.sum((not_rated > 0).astype(np.int32), axis=1)
    pick_probs = not_rated.astype(
        np.int32) / np.sum(not_rated.astype(np.int32), axis=1)[:, None]
    for user in range(train_rating_matrix.shape[0]):
        negatives = np.random.choice(
            items, size=min(n_not_rated[user], n_test_rated[user] * n_negatives),  replace=False, p=pick_probs[user])
        mask[user, negatives] = 1
    return mask

def ranking_list_from_mask(mask, n_users, n_negatives):

    y_shape = np.max(np.sum((mask==0).astype(np.int32),axis=1))
    rank_list = np.zeros([n_users, n_negatives+1])
    out = rank_list.copy()
    for u in range(n_users):
        items = np.argwhere(mask[u]!=0).flatten()
        rank_list[u,:len(items)] = items

    return rank_list.astype(NUMPY_INT_DTYPE), out.astype(NUMPY_FLOAT_DTYPE)



def get_HR_ranking_list(train_rating_matrix, test_rating_matrix, test_df, n_negatives=50):

    #mask = test_rating_matrix > 0
    not_rated = (train_rating_matrix + test_rating_matrix) == 0
    items = np.arange(train_rating_matrix.shape[1])
    n_test_rated = np.sum((test_rating_matrix > 0).astype(np.int32), axis=1)
    n_not_rated = np.sum((not_rated > 0).astype(np.int32), axis=1)#
    pick_probs = not_rated.astype(
        np.int32) / np.sum(not_rated.astype(np.int32), axis=1)[:, None]
    ranking_list = np.zeros([train_rating_matrix.shape[0], n_negatives+1]).astype(np.int32)
    ranking_list[test_df['user_id'],0] = test_df['item_id']
    for user in range(train_rating_matrix.shape[0]):
        negatives = np.random.choice(
            items, size=min(n_not_rated[user], n_test_rated[user] * n_negatives),  replace=False, p=pick_probs[user])
        ranking_list[user,1:] = negatives
    return ranking_list




def calculate_metrics(recommendations_list, actual_rating_matrix, get_average=False):
    
    max_hits = np.sum(actual_rating_matrix > 0)
    rateable = np.count_nonzero(np.sum(actual_rating_matrix, axis=1))
    top_n = recommendations_list.shape[1]
    tpfn = rateable * top_n
    
    true_positives = 0
    __ux=[]
    for user in range(recommendations_list.shape[0]):
        _ux = 0
        for item in recommendations_list[user]:
            if actual_rating_matrix[user, item] > 0:
                true_positives += 1
                _ux += 1
        if get_average:
            __ux.append(_ux/recommendations_list.shape[1])
    precision = true_positives / tpfn
    recall = true_positives / max_hits

    result = {"PRECISION@{}".format(
        top_n): precision, "RECALL@{}".format(top_n): recall}
    if not get_average:
        return result
    return result, __ux

def evaluate_ranking(train_rating_matrix, test_rating_matrix, predicted_matrix, top_n=10, n_negatives=50):
    mask = np.array(partial_ranking_mask(train_rating_matrix,
                                         test_rating_matrix, n_negatives=n_negatives))
    predicted_matrix = np.array(predicted_matrix.todense())
    predicted_matrix[mask == 0] = -np.inf
    top_n_recommendations = np.argsort(predicted_matrix, axis=1)[:, -top_n:]
    return calculate_metrics(top_n_recommendations, np.array(test_rating_matrix.todense()))



def most_pop_recommender(dataset, topN = 10):

    R=(dataset.train_rating_matrix > 0).astype(np.int32)
    mask = partial_ranking_mask(dataset.train_rating_matrix, dataset.test_rating_matrix)
    summed = np.sum(R, axis=0)
    rec=np.zeros(R.shape)
    rec+=summed
    rec[mask==0]=-np.inf
    rec[dataset.train_rating_matrix>0]=-np.inf
   
    rec_list = np.argsort(rec, axis=1)[:,-topN:]
    print(calculate_metrics(rec_list, dataset.test_rating_matrix))

