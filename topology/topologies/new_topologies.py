import numpy as np
from topology.properties import *
from topology.cython.top_funcs import update_model_counts

class BaseTopology():


    def __init__(self):

        pass

    def fit(self, dataset):

        self._fit(dataset)


    def update(self, sim):

        self._update()

    def _fit(self, dataset, extra_nodes=0):

        self.n_users = dataset.n_users
        self.current_G = np.zeros([dataset.n_users+extra_nodes, dataset.n_users+extra_nodes]).astype(int_dtype)
        self.base_G = np.zeros([dataset.n_users+extra_nodes, dataset.n_users+extra_nodes]).astype(int_dtype)
        self.is_aggregator = np.zeros([dataset.n_users+extra_nodes]).astype(int_dtype)
        self.has_model = np.zeros([dataset.n_users+extra_nodes]).astype(int_dtype)
        self.has_data = np.zeros([dataset.n_users+extra_nodes]).astype(int_dtype)
        self.n_nodes = self.current_G.shape[0]
        self.model_counts = np.zeros(self.current_G.shape[0]).astype(np.int32)
        self.walk_step=0
        self.extra_Graph = self.current_G.copy()
        self.aggregate_single_update=0
        
    def _update(self):
 
        
        if self.walk_step==0:
          
            self.current_G[self.samples_array[:,0],self.samples_array[:,1]] = 1
            update_model_counts(self.model_counts, self.samples_array[:,1], None, self.extra_Graph)
            self.walk_step+=1
        
        elif self.walk_step==self.samples_array.shape[1]-1:
            self.current_G[self.samples_array[:,self.walk_step-1],self.samples_array[:,self.walk_step]] = 0
            self.old_out_nodes = self.samples_array[:,-1].copy()
            self.sample()
            update_model_counts(self.model_counts, self.samples_array[:,1], self.samples_array[:,0],self.extra_Graph)
            self.current_G[self.samples_array[:,0],self.samples_array[:,1]] =1
            self.walk_step=1

        
        else:
            self.current_G[self.samples_array[:,self.walk_step-1],self.samples_array[:,self.walk_step]] = 0
            
            self.current_G[self.samples_array[:,self.walk_step],self.samples_array[:,self.walk_step+1]] = 1
            update_model_counts(self.model_counts,self.samples_array[:,self.walk_step+1],self.samples_array[:,self.walk_step],self.extra_Graph)
            self.walk_step+=1
           
        #
      
    def _sample(self):

        pass


class FederatedTopology(BaseTopology):

    
    def __init__(self, K=0.1):

        self.K = K
        self.counter=0
        super( FederatedTopology, self).__init__()


    def _fit(self, dataset):
        
        super()._fit(dataset, 1)
        self.user_ids = np.arange(dataset.n_users)
        self.is_aggregator[-1] = 1
        #self.has_model[-1] = 1
        self.has_data[:self.n_users] = 1
        self.seeds = [self.n_users] 
        self.valid_attack_nodes = np.zeros(self.current_G.shape[0])
        self.valid_attack_nodes[-1] = 1
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes[-1] = 0
        self.sample()
        self._update()
    
    def sample(self):

        sampled = np.random.choice(self.user_ids, size=int(self.K*len(self.user_ids)), replace=False).reshape(-1,1)
     
        ones = np.full([len(sampled),1],self.current_G.shape[0]-1,dtype=np.int32)
        self.samples_array=np.concatenate(
            [
                ones.copy(),
                sampled,
                ones.copy()
            ],
            axis=1
        )
        
class FixedHierarchicalFederatedLearningTopology(BaseTopology):

    """
    Splits users into subgroups, with a server sampling each of them.
    """

    def __init__(self, n_sub_servers=3, K=0.1):

        self.n_sub_servers=3
        self.K=K


    def _fit(self, dataset):
        
        super()._fit(dataset, 1+self.n_sub_servers)
        self.user_ids = np.arange(dataset.n_users)
        self.is_aggregator[-(1+self.n_sub_servers):] = 1
        #self.has_model[-1] = 1
       
        self.subservers = np.arange(self.n_sub_servers) + dataset.n_users
        self.valid_attack_nodes = np.zeros(self.current_G.shape[0])
        self.valid_attack_nodes[self.subservers] = 1
        self.has_data[:self.n_users] = 1
        self.seeds = [self.current_G.shape[0]-1] 
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes[dataset.n_users:] = 0
        self.sample()
        self._update()

    def sample(self):
        
        out=[]
        for i in range(len(self.subservers)):
            sample = np.random.choice(self.user_ids,
                                size=int(len(self.user_ids)*self.K*(1/self.n_sub_servers)),
                                replace=False).reshape(-1,1)

            ones = np.full([len(sample),1],self.current_G.shape[0]-1)
            sub = np.full([len(sample),1], self.subservers[i])
  
            
            array = np.concatenate([
                ones.copy(),
                sub.copy(),
                sample,
                sub.copy(),
                ones.copy()

            ], axis=1)
            out.append(array)
        self.samples_array = np.concatenate(out, axis=0)



class RandomWalkTopology(BaseTopology):
    """
    Samples n walks of fixed length

    """

    def __init__(self, n_walks=10, walk_length=10):

        self.n_walks = n_walks
        self.walk_length = walk_length
        self.counter=0
        super( RandomWalkTopology, self).__init__()
        pass

    def _fit(self, dataset):
        
        super()._fit(dataset, 1)
        self.user_ids = np.arange(dataset.n_users)
        self.is_aggregator[-1] = 1
        self.has_data[:self.n_users] = 1
        self.seeds = [self.n_users] 
        self.valid_attack_nodes = np.ones(self.current_G.shape[0])
        self.valid_attack_nodes[-1] = 1
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes[dataset.n_users:] = 0
        self.sample()
        self._update()

    def sample(self):

        self.samples_array = np.random.randint(0, len(self.user_ids),[self.n_walks, self.walk_length+2])
        self.samples_array[:,0] = self.current_G.shape[0]-1
        self.samples_array[:,-1] = self.current_G.shape[0]-1





class ServerlessRandomWalkTopology(BaseTopology):

    def __init__(self, density=0.01, n_aggregators=0):

        self.density=density
        self.n_aggregators=n_aggregators
        super(ServerlessRandomWalkTopology, self).__init__()
        pass

    def _fit(self, dataset):
        
        super()._fit(dataset, self.n_aggregators)
        self.user_ids = np.arange(dataset.n_users)
        self.is_aggregator[-self.n_aggregators:] = 1
        self.has_data[:self.n_users] = 1
        self.two_way_node = np.zeros(self.has_data.shape).astype(int_dtype)
        self.two_way_node[-self.n_aggregators:] = 1
        mask = np.random.random(self.current_G.shape) < self.density
      
        #self.has_model[self.sampled[:,0]] = 1
        self.valid_attack_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
        self.old_end_nodes = None
        self.sample()
        self._update()
        self.seeds = self.samples_array[:,0]


    def sample(self):

        self.samples_array = np.random.randint(0, len(self.user_ids), [int(len(self.user_ids)*self.density),10])
        if self.old_end_nodes is not None:
            self.samples_array[:,0] = self.old_end_nodes

        self.old_end_nodes = self.samples_array[:,-1].copy()



class GLTopology(ServerlessRandomWalkTopology):

    def __init__(self, density=0.01, n_aggregators=0):

        super(GLTopology, self).__init__(density, n_aggregators)
    
    def _fit(self, dataset):

        super()._fit(dataset)
        #every node starts out with model
        self.seeds = np.arange(self.current_G.shape[0])
        self.aggregate_single_update=1


class ConsensusNodesTopology(FixedHierarchicalFederatedLearningTopology):
    
    def __init__(self, n_consensus_servers=3, n_walks=10, walk_length=10):

        self.n_sub_servers=n_consensus_servers
        self.n_walks=n_walks
        self.walk_length=walk_length
        

    def _fit(self, dataset):
        self.stage=0
        super()._fit(dataset)
        

    def sample(self):

        if self.stage==0:
            subservers = np.repeat(self.subservers, self.n_walks)
            self.samples_array = np.random.randint(0,len(self.user_ids),[len(self.subservers)*self.n_walks, self.walk_length])
            self.samples_array[:,0] = subservers.copy()
            self.samples_array[:,-1] = subservers.copy()
            self.stage=1

        else:
            subservers = np.repeat(self.subservers, len(self.subservers))
            subservers_b = np.tile(self.subservers, len(self.subservers))
            self.samples_array = np.zeros([len(subservers),3]).astype(np.int64)
            self.samples_array[:,0] = subservers
            self.samples_array[:,1] = subservers_b
            self.samples_array[:,2] = subservers
            self.stage=0

            






      









            

