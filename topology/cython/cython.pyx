import  numpy as np
cimport numpy as np
from cython import boundscheck
from cython.parallel cimport prange
from libc.math cimport fabs as c_abs
from libc.math cimport pow, exp, log
from libc.stdio cimport printf
from libc.stdlib cimport rand
from libc.string cimport memcpy, memset
string = 'my_string23'

def aggregate_nodes(np.ndarray[np.float64_t, ndim=3] Q_old,
                    np.ndarray[np.float64_t, ndim=3] Q_new,
                    np.ndarray[np.float64_t, ndim=3] workspace,
                    np.ndarray[np.int32_t, ndim=2] counts,
                    np.ndarray[np.int32_t, ndim=2] sample_matrix,
                    np.ndarray[np.int32_t, ndim=1] model_counts,
                    np.ndarray[np.int32_t, ndim=1] aggregators,
                    np.ndarray[np.int32_t, ndim=2] G,
                    int num_threads=70,
                    int verbose=0):



    
    
    cdef int i,j, k, agg, n_aggregators, had_model
    cdef int n_items = Q_old.shape[1]
    cdef int n_factors = Q_old.shape[2]
    cdef np.ndarray[np.float64_t, ndim=1] count = np.zeros(aggregators.shape[0]).astype(np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] has_model = np.zeros([aggregators.shape[0]]).astype(np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] zeros = np.zeros([Q_new.shape[1], Q_new.shape[2]]).astype(np.float64)
    with nogil, boundscheck(True):

        for agg in prange(aggregators.shape[0],num_threads=num_threads):
            
            if model_counts[agg] > 1:
                #printf("Aggregated!\n")
                has_model[agg]=1
                count[agg]=0
                for i in range(G.shape[0]):
                    
                    if G[i,agg] > 0:
                        #printf("Aggregating\n")
                        count[agg]+=1.0
                     
                        for k in range(n_items):
                            for j in range(n_factors):
                                workspace[agg, k, j] += Q_old[i,k,j]
                
                for i in range(n_items):
                    for j in range(n_factors):
                        Q_new[agg,i,j] = workspace[agg,i,j] / count[agg]
                        workspace[agg,i,j]=0
            elif model_counts[agg]>0:
                
                had_model=0
                for i in range(G.shape[1]):
                

                    if G[i, agg] > 0:
                        
                        memcpy(&Q_new[agg,0,0],&Q_old[i,0,0],sizeof(np.float64_t)*n_items*n_factors)
                        had_model=1
                        has_model[agg] = 1
                        
                        break

            else:
                memcpy(&Q_new[agg,0,0],&Q_old[agg,0,0],sizeof(np.float64_t)*n_items*n_factors)


               
    if verbose:

        print("N aggregations:",len(np.argwhere(count)>1), 'total-summed',count.sum())
        #input()
    return Q_new,Q_old, has_model

def aggregate_nodes_new(np.ndarray[np.float64_t, ndim=3] Q1,
                    np.ndarray[np.float64_t, ndim=3] Q2,
                    np.ndarray[np.float64_t, ndim=3] workspace,
                    np.ndarray[np.int32_t, ndim=2] counts,
                    np.ndarray[np.int32_t, ndim=2] sample_matrix,
                    np.ndarray[np.int32_t, ndim=1] model_counts,
                    np.ndarray[np.int32_t, ndim=1] aggregators,
                    np.ndarray[np.int32_t, ndim=2] G,
                    int num_threads=70,
                    int verbose=0):



    
    
    cdef int i,j, k, agg, n_aggregators, had_model
    cdef int n_items = Q1.shape[1]
    cdef int n_factors = Q1.shape[2]
    cdef np.ndarray[np.float64_t, ndim=1] count = np.zeros(aggregators.shape[0]).astype(np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] has_model = np.zeros([aggregators.shape[0]]).astype(np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] zeros = np.zeros([Q1.shape[1], Q1.shape[2]]).astype(np.float64)
    with nogil, boundscheck(True):

        for agg in prange(aggregators.shape[0],num_threads=num_threads):
            
            if model_counts[agg] > 1:
                #printf("Aggregated!\n")
                has_model[agg]=1
                count[agg]=0
                for i in range(G.shape[0]):
                    
                    if G[i,agg] > 0:
                        #printf("Aggregating\n")
                        count[agg]+=1.0
                     
                        for k in range(n_items):
                            for j in range(n_factors):
                                workspace[agg, k, j] += Q1[i,k,j]
                
                for i in range(n_items):
                    for j in range(n_factors):
                        Q2[agg,i,j] = workspace[agg,i,j] / count[agg]
                        workspace[agg,i,j]=0
            elif model_counts[agg]>0:
                
                had_model=0
                for i in range(G.shape[1]):
                

                    if G[i, agg] > 0:
                        
                        memcpy(&Q2[agg,0,0],&Q1[i,0,0],sizeof(np.float64_t)*n_items*n_factors)
                        had_model=1
                        has_model[agg] = 1
                        
                        break
        
        for agg in prange(aggregators.shape[0],num_threads=num_threads):
            if model_counts[agg] > 0:
                memcpy(&Q1[agg,0,0],&Q2[agg,0,0],sizeof(np.float64_t)*n_items*n_factors)

           


               
    if verbose:

        print("N aggregations:",len(np.argwhere(count)>1), 'total-summed',count.sum())
        #input()
    return Q1,Q2, has_model


def aggregate_nodes_smart(np.ndarray[np.float64_t, ndim=3] Q1,
                    np.ndarray[np.float64_t, ndim=3] Q2,
                    np.ndarray[np.float64_t, ndim=3] workspace,
                    np.ndarray[np.int32_t, ndim=3] counts,
                    np.ndarray[np.int32_t, ndim=2] sample_matrix,
                    np.ndarray[np.int32_t, ndim=1] model_counts,
                    np.ndarray[np.int32_t, ndim=1] aggregators,
                    np.ndarray[np.int32_t, ndim=2] G,
                    int num_threads=70,
                    int verbose=0,
                    int aggregate_single_update=0):



    
    
    cdef int i,j, k, agg, n_aggregators, had_model
    cdef int n_items = Q1.shape[1]
    cdef int n_factors = Q1.shape[2]
    cdef np.ndarray[np.float64_t, ndim=1] count = np.zeros(aggregators.shape[0]).astype(np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] has_model = np.zeros([aggregators.shape[0]]).astype(np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] zeros = np.zeros([Q1.shape[1], Q1.shape[2]]).astype(np.float64)
    with nogil, boundscheck(True):

        for agg in prange(aggregators.shape[0],num_threads=num_threads):
            
            if model_counts[agg] > 1:
                has_model[agg]=1
                if aggregate_single_update==0:
                    
                    for k in range(G.shape[0]):

                        if G[k, agg] > 0:
                            #printf("Aggregating %d %d\n", agg, k)
                            for i in range(n_items):
                                
                                    
                                    for j in range(n_factors):
                                        if Q1[agg,i,j] != Q1[k,i,j]:
                                            counts[agg,i,j]+=1  
                                            workspace[agg,i,j] += Q1[k,i,j]  - Q1[agg,i,j]
                    
                    for i in range(n_items):
                        
                        for j in range(n_factors):
                            if counts[agg, i,j]>0:
                                Q2[agg,i,j]=Q1[agg,i,j] + (workspace[agg,i,j]/counts[agg,i,j])
                                workspace[agg,i,j]=0
                                counts[agg,i,j]=0
                            else:
                        
                                Q2[agg,i,j] = Q1[agg,i,j]
                else:
                    memcpy(&Q2[agg,0,0], &Q1[agg,0,0],sizeof(np.float64_t)*n_factors*n_factors)
                    count[agg]=1
                    for k in range(G.shape[0]):

                        if G[k, agg] > 0:
                            count[agg]+=1
                            for i in range(n_items):
                                for j in range(n_factors):
                                    Q2[agg,i,j] += Q1[k,i,j]
                    for i in range(n_items):
                        for j in range(n_factors):
                            Q2[agg,i,j] = Q2[agg,i,j] / count[agg]
                    
            elif model_counts[agg]==1:
                has_model[agg]=1
                if aggregate_single_update==0:
                
                    for k in range(G.shape[0]):
                        if G[k,agg]>0:
                  
                            memcpy(&Q2[agg,0,0],&Q1[k,0,0], sizeof(np.float64_t)*n_factors * n_items)
                            
                            break
                else:
                    for k in range(G.shape[0]):

                        if G[k,agg]>0:

                            for i in range(n_items):
                                for j in range(n_factors):

                                    Q2[agg,i,j] = (Q1[agg,i,j] + Q1[k,i,j]) / 2

                            break
                
        for agg in prange(aggregators.shape[0], num_threads=num_threads):#,num_threads=num_threads):
            if model_counts[agg]>0:
                memcpy(&Q1[agg,0,0], &Q2[agg,0,0], sizeof(np.float64_t)*n_items*n_factors)
         
    if verbose:

        print("N aggregations:",len(np.argwhere(count)>1), 'total-summed',count.sum())
        #input()
    return Q1,Q2, has_model







def update_model_counts(np.ndarray[np.int32_t, ndim=1] model_counts,
                        np.ndarray[np.int64_t, ndim=1] sampled,
                        np.ndarray[np.int64_t, ndim=1] unsampled,
                        np.ndarray[np.int32_t, ndim=2] G):

    cdef int i
    if unsampled is not None:
        for i in range(unsampled.shape[0]):

            model_counts[unsampled[i]]=0
        for i in range(sampled.shape[0]):
            if G[unsampled[i],sampled[i]]==0:
                model_counts[sampled[i]] += 1
                G[unsampled[i],sampled[i]] =1 

        for i in range(sampled.shape[0]):
            G[unsampled[i], sampled[i]] = 0
        
    else:
        for i in range(sampled.shape[0]):
            model_counts[sampled[i]] += 1
    
def bpr_multi_update( 
                        np.ndarray[np.float64_t, ndim=2] U not None,
                        np.ndarray[np.float64_t, ndim=3] I not None,
                        np.ndarray[np.int32_t, ndim=2] users not None,
                        np.ndarray[np.int32_t, ndim=2] pos_items not None,
                        np.ndarray[np.int32_t, ndim=2] neg_items not None,
                        np.ndarray[np.int32_t, ndim=1] n_samples not None,
                        np.ndarray[np.float64_t, ndim=1] diffs not None,
                        np.ndarray[np.float64_t, ndim=1] losses not None,
                        np.ndarray[np.int32_t, ndim=1] has_model,
                        np.ndarray[np.int32_t, ndim=1] has_data,
                        np.float64_t lr,
                        np.float64_t reg,
                        int num_threads
                       
                        ):

    cdef Py_ssize_t user_id, i,j,u,idx, jdx, n_factors, node_id
    cdef double pos_predict,noise_u,noise_i, noise_j, neg_predict,grad_i, grad_j, grad_u, u_factor, i_factor, j_factor, sigmoid
    
    n_factors = I.shape[2]
    
    with nogil,  boundscheck(True):
        
        for node_id in prange(I.shape[0],num_threads=num_threads):

            if has_model[node_id]==0 or has_data[node_id]==0:
                continue
         
            losses[node_id] = 0

            for idx in range(0,n_samples[node_id]):

                diffs[node_id] = 0
                u = users[node_id, idx]
                i = pos_items[node_id, idx]
                j = neg_items[node_id, idx]

                for jdx in range(n_factors):
                   
                    diffs[node_id] += (U[u,jdx] * I[u, i, jdx]) - (U[ u,jdx] * I[u, j, jdx])
                
              
                losses[u]+=-log(1 / (1 + exp(-diffs[u])))
                diffs[u] = 1 / (1 + exp(diffs[u]))
            
                for jdx in range(n_factors):

                    u_factor = U[u, jdx] + 0
                    i_factor = I[u, i, jdx] + 0
                    j_factor = I[u, j, jdx] + 0

                    U[ u, jdx] += lr * (diffs[node_id] * (i_factor - j_factor) - (reg * u_factor))

                    I[u, i, jdx] += lr * (diffs[node_id] *  u_factor - (reg * i_factor ))

                    I[u, j, jdx] += lr *  ( (diffs[node_id] * (- u_factor)) - (reg * j_factor ))

            losses[u] = losses[u] / n_samples[u]


    return U, I, losses


def evaluate_nodes(np.ndarray[np.float64_t, ndim=2] U,
                    np.ndarray[np.float64_t, ndim=3] I,
                    np.ndarray[np.int32_t, ndim=2] items,
                    np.ndarray[np.float64_t, ndim=2] scores,
                    np.ndarray[np.float64_t, ndim=2] test_rating_matrix,
                    int num_threads,
                    int topN,
                    ):

    cdef Py_ssize_t user_id, idx, n_factors,j,item
    cdef double total
    cdef np.ndarray[dtype=np.float64_t, ndim=1] recall
    n_factors = I.shape[2]
    n_users = U.shape[0]
    recall = np.zeros(n_users, dtype=np.float64)


    with nogil,  boundscheck(True):

        for user_id in prange(U.shape[0]):

            for idx in range(items.shape[1]):
                #printf("%d, %d\n",user_id, idx)
                

                scores[user_id, idx] = 0
                item = items[user_id, idx]
                for j in range(n_factors):
                    scores[user_id, idx] += U[user_id, j] * I[user_id,item, j]
    #how to make this run in parallel???
    sorted_indices = np.argsort(scores,axis=1)
    cdef np.ndarray[np.int32_t, ndim=2] rec_list = items[np.arange(items.shape[0])[:,None], sorted_indices][:,-topN:]
    #return rec_list
    cdef int start_idx = rec_list.shape[1] - topN
    
    with nogil, boundscheck(False):

        for user_id in prange(rec_list.shape[0], num_threads=num_threads):


            for idx in range(0, rec_list.shape[1]):

                if test_rating_matrix[user_id, rec_list[user_id,idx]] > 0:

                    recall[user_id] = 1


    return recall.mean()


    return rec_list












    

