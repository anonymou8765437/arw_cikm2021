import numpy as np
from topology.cython import top_funcs
from topology.properties import *
#from topology.topologies.topologies import *
from topology.utils.data_processing import *
from topology.utils.evaluation import *
from topology.privacy.PrivacyModule import *

class TopologySim():


    def __init__(self,
                lr=0.05,
                reg=0.01,
                confidence=1.0,
                n_negatives=5,
                topN=10,
                E=5,
                n_test_negatives=50,
                num_threads=72,
                n_factors=5,
                max_steps=1000,
                update_prob=1.0,
                evaluation_step=1,
                init='normal',
                validate=False,
                init_mean=0,
                init_std=0.01,
                privacy_module=None,
                corr_module=None,
                verbose=True,
                ):
        
        self.privacy_module = privacy_module
        self.evaluation_step=evaluation_step
        self.update_prob = update_prob
        self.num_threads=num_threads
        self.verbose=verbose
        self.lr=lr
        self.E=E
        self.topN=topN
        self.max_steps = max_steps
        self.reg=reg
        self.n_factors=n_factors
        self.confidence=confidence
        self.n_negatives=n_negatives
        self.n_test_negatives=n_test_negatives
        self.init=init
        self.init_mean = init_mean
        self.init_std = init_std
        self.validate=validate
        self.corr_module = corr_module
    def fit(self, 
            dataset,
            topology=None):
        print("Processing data")
        self.dataset = dataset
        if isinstance(self.privacy_module ,FeatureInferenceModule):
            self.privacy_module.pre_fit(self)

        self.process_data(dataset)
        print("Processing topology")
        self.process_topology(topology)
        print("initializing variables")
        self._init_vars()
        
        return self._fit()        
    
    #@profile  
    def _fit(self):
        #initial propagation

        if self.privacy_module is not None:
            self.privacy_module.fit(self)

        if self.corr_module is not None:
            self.corr_module.fit(self)
        
        
        for self.step in range(self.max_steps):
         
            try:

                self._step()
                self._evaluate()
               
            except KeyboardInterrupt:
                return self.get_results()

        return self.get_results()

    def get_train_df(self):
        return create_negative_training_samples_bpr(self.dataset.train_rating_matrix,n_negatives=self.n_negatives)

    def get_sample_matrix(self):

        X = np.zeros(self.dataset.train_rating_matrix.shape).astype(int_dtype)
        df=self.train_df
        X[df['user_id'],df['item_id_i']] = 1
        X[df['user_id'],df['item_id_j']] = 1
        return X

    def df_to_lists(self):
        
        self.users = np.zeros([self.n_nodes, self.max_samples]).astype(np.int32)
        self.item_i = self.users.copy()
        self.item_j = self.item_i.copy()
        
        for u, df in self.train_df.groupby('user_id'):
            self.users[u,:len(df)] = df['user_id'].values
            self.item_i[u,:len(df)] = df['item_id_i'].values
            self.item_j[u,:len(df)] = df['item_id_j'].values



    def process_data(self, dataset):
        self.dataset=dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.train_df = self.get_train_df()
        self.sample_matrix = self.get_sample_matrix()
        self.n_samples = np.array([
            len(df) for u, df in self.train_df.groupby('user_id')
        ]).astype(int_dtype)
        self.max_samples = self.n_samples.max()
        self.get_mask()
        self.get_ranking_list()

    def get_ranking_list(self):

        self.ranking_list = get_HR_ranking_list(
            self.dataset.train_rating_matrix,
            self.dataset.test_rating_matrix,
            self.dataset.test_df,
            self.n_test_negatives
        )
        #self.ranking_list,out = ranking_list_from_mask(self.mask, self.mask.shape[0],self.n_test_negatives)
        self.scores = np.zeros(self.ranking_list.shape).astype(np.float64)

    def get_mask(self):

        if self.n_test_negatives is None:
            self.mask = None

        else:
            self.mask = partial_ranking_mask(self.dataset.train_rating_matrix,
                                            self.dataset.test_rating_matrix,
                                            self.n_test_negatives)
        
    def _init_vars(self):

        self.recall=[]
        self.data_counter=0
        self.data_sent=[]
        self.P = np.zeros([self.n_users, self.n_factors])
        self.Q1 = np.zeros([self.n_nodes, self.n_items, self.n_factors])
        self.Q2 = self.Q1.copy()
        self.agg_counts = np.zeros([self.n_nodes, self.n_items, self.n_factors]).astype(np.int32)
        self.data = np.zeros([self.n_users,3, self.max_samples])
        self.aggregators = self.topology.is_aggregator
        self.agg_space = np.zeros(self.Q1.shape)
        self.model_counts = np.zeros(self.n_nodes).astype(int_dtype)
        self.propagated = np.zeros(self.n_nodes).astype(np.int32)
        self.df_to_lists()
        self.loss = np.zeros(self.aggregators.shape).astype(np.float64)
        self.diffs = np.zeros(self.aggregators.shape).astype(np.float64)
        self.has_model = self.topology.has_model
        self.prev_has_model = np.zeros(self.has_model.shape).astype(int_dtype)
        self.sent = np.zeros(self.G.shape).astype(np.int32)
        self.seed_models()

    def seed_models(self):

        #put initial model into the network
        
        if self.init=='normal':
            P = np.random.normal(self.init_mean, self.init_std,[self.n_users, self.n_factors])
            Q = np.random.normal(self.init_mean, self.init_std, [self.n_items, self.n_factors])
        
        else:
            P = np.random.random([self.n_users, self.n_factors])
            Q = np.random.random([self.n_items, self.n_factors])
        

        self.P[:] = P
        self.Q1[self.topology.seeds,:,:] = Q
    
    
    #@profile
    def _step(self):
        
        if self.privacy_module is not None and self.step != 0 and self.step % self.evaluation_step==0:
            self.privacy_module.attack()
        if self.corr_module is not None and self.step != 0 and self.step % self.evaluation_step==0:
            self.corr_module.evaluate()

        self._aggregate()
        self._train()
        self._update_topology()
 

    def process_topology(self, 
                        topology):
        self.topology=topology
        self.topology.fit(self.dataset)
        self.n_nodes = topology.n_nodes
        self.has_data = topology.has_data
        self.G = self.topology.current_G
        self.user_ids = self.topology.user_ids

    def _update_topology(self):
        self.topology.update(self)
    #@profile
    def _aggregate(self):
        
        if self.validate:
            old = self.Q1.copy()

        self.Q1, self.Q2, self.has_model = top_funcs.aggregate_nodes_smart(
            self.Q1, 
            self.Q2, 
            self.agg_space,
            self.agg_counts, 
            self.sample_matrix,
            self.topology.model_counts, 
            self.aggregators,
            self.G, 
            self.num_threads,
            self.topology.aggregate_single_update
        )

        if self.validate and self.step>0:
            #ensure correct nodes have been aggregated
            new = self.Q1
            X= (new-old).sum(axis=(1,2))
           
            a=np.argwhere(X!=0).flatten()
            b = np.argwhere(self.model_counts>0).flatten()
            #print(len(a), len(np.argwhere(self.topology.model_counts)>0))
            #assert len(a) == len(np.argwhere(self.topology.model_counts>0))
            if len(a) != len(np.argwhere(self.topology.model_counts>0)) or (len(np.setdiff1d(a,b))!=0 or len(np.setdiff1d(b,a)) != 0 ):
                
                invalid = np.setdiff1d(b,a)
                for m in invalid: 

                    neigh = np.argwhere(self.G[:,m]>0).flatten()
                    #if a node's only neighbor is itself then an error is to be expected
                    if len(neigh) == 1 and neigh[0]==m:

                        pass
                    else:
                        raise ValueError('Un aggregated nodes')

            #ensure correct aggregation
            for idx, m in enumerate(a):
                neigh = np.argwhere(self.G[:,m]>0).flatten()
                
                if len(neigh)==1:
                    X = self.Q1[m] - old[neigh[0]]
                    try:
                        assert X.sum() == 0
                    except:
                       
                        # for t in np.argwhere(X!=0):

                        #     print(X[t[0],t[1]], self.Q1[m,t[0],t[1]])
                        raise ValueError("Miscopied nodes")
                else:
                    msk = old[neigh] - old[m]
                    p = (msk!=0).sum(axis=0).astype(np.int32)
                    p[p==0]=1
                    X = np.sum(msk,axis=0) / p
                    X = old[m] + X
                    nn = set(list(neigh))
                
                    try:
                        assert (self.Q1[m] - X).sum() == 0
                    except:
                        
                        raise ValueError("Misaggregated nodes")

             


    def _train(self):

        #for ARW etc, get probability of actually updating the model..
        update_probs = self.has_data * (np.random.random(self.has_data.shape)<self.update_prob).astype(np.int32)
        
        if self.validate:
            old = self.Q1.copy()
        # print(np.argwhere(self.has_model > 0))
        # input()
        for e in range(self.E):
            
            self.P, self.Q1, loss = top_funcs.bpr_multi_update(
                self.P,
                self.Q1,
                self.users,
                self.item_i,
                self.item_j,
                self.n_samples,
                self.diffs,
                self.loss,
                self.has_model,
                update_probs,
                self.lr,
                self.reg,
                self.num_threads,
    
            )

        if self.validate and self.step>0:
            
            X = (self.Q1 - old).sum(axis=(1,2))
            a = np.argwhere(X!=0).flatten()
            b=np.argwhere((update_probs*self.has_model)>0).flatten()
            assert len(np.setdiff1d(a,b))==0 and len(np.setdiff1d(b,a))==0
        
 
       
    def get_results(self):

        x= {
            'recall':self.recall,
            'data-sent':self.data_sent
        }

        if self.privacy_module is not None:
            x['attack-success'] = self.privacy_module.get_results()

        if self.corr_module is not None:
            x['distance-correlation'] = self.corr_module.history

        return x
    
    #@profile
    def _evaluate(self):
        
        self.data_counter += self.topology.model_counts.sum()
        if self.step > 0 and self.step % self.evaluation_step == 0:
          
            self.data_sent.append(self.data_counter)
            # Q = self.Q1[self.user_ids]
            # P = self.P
            # pred = np.einsum( "ij,ikj->ik", P, Q)
            # if self.mask is not None:
            #     pred*=self.mask
            # pred[self.dataset.train_rating_matrix>0]=-np.inf
            # rec = np.argsort(pred,axis=1)[:,-self.topN:]
            # x=calculate_metrics( rec, self.dataset.test_rating_matrix)
            # #print(x['RECALL@10'])
            # print(self.ranking_list.shape, self.scores.shape)
            y = top_funcs.evaluate_nodes(self.P,
                    self.Q1,
                    self.ranking_list,
                    self.scores,
                    self.dataset.test_rating_matrix,
                    self.num_threads,
                    self.topN)
            
            self.recall.append(y)
            if self.verbose:
                print("Recall:",y)
            #input()
    






