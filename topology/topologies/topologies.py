import numpy as np
from topology.properties import *
from topology.cython.top_funcs import update_model_counts

class BaseTopology():


    def __init__(self):

        pass

    def fit(self, dataset):

        self._fit(dataset)


    def update(self, sim):

        self._update(sim)

    def _fit(self, dataset, extra_nodes=0):

        self.n_users = dataset.n_users
        self.current_G = np.zeros([dataset.n_users+extra_nodes, dataset.n_users+extra_nodes]).astype(int_dtype)
        self.base_G = np.zeros([dataset.n_users+extra_nodes, dataset.n_users+extra_nodes]).astype(int_dtype)
        self.is_aggregator = np.zeros([dataset.n_users+extra_nodes]).astype(int_dtype)
        self.has_model = np.zeros([dataset.n_users+extra_nodes]).astype(int_dtype)
        self.has_data = np.zeros([dataset.n_users+extra_nodes]).astype(int_dtype)
        self.n_nodes = self.current_G.shape[0]
        self.model_counts = np.zeros(self.current_G.shape[0]).astype(np.int32)

        
    def _update(self, sim):

        self.walk_step+=1 
        if self.walk_step>self.samples_array.shape[1]-2:
            self.current_G[self.samples_array[:,self.walk_step-1],self.samples_array[:,self.walk_step]] = 0
            self.sample()
        else:
            self.current_G[self.samples_array[:,self.walk_step-1],self.samples_array[:,self.walk_step]] = 0
            
            self.current_G[self.samples_array[:,self.walk_step],self.samples_array[:,self.walk_step+1]] = 1
            update_model_counts(self.model_counts,self.samples_array[:,self.walk_step+1],self.samples_array[:,self.walk_step])


    def _sample(self):

        pass


        


class FederatedStarTopology(BaseTopology):

    def __init__(self, K=0.1):

        self.K = K
        self.counter=0
        super( FederatedStarTopology, self).__init__()


    def _fit(self, dataset):
        
        super()._fit(dataset, 1)
        self.user_ids = np.arange(dataset.n_users)
        self.is_aggregator[-1] = 1
        self.has_model[-1] = 1
        self.sampled= np.random.choice(np.arange(self.n_users),size=int(self.K*len(self.user_ids)),replace=False)
        self.current_G[-1,self.sampled] = 1
        self.has_data[:self.n_users] = 1
        self.seeds = [self.n_users] 
        self.two_way_node = np.ones(self.current_G.shape[0]).astype(int_dtype) 
        self.valid_attack_nodes = np.zeros(self.current_G.shape[0])
        self.valid_attack_nodes[-1] = 1
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes[-1] = 0
        self.step=1
    
    def _update(self, sim):
        
        self.step = (self.step+1)%2

        if self.step==0:

            self.current_G[-1, self.sampled]=0
            self.current_G[self.sampled, -1] = 1
            update_model_counts(self.model_counts, np.full(len(self.sampled),-1),self.sampled)
        else:
            self.current_G[self.sampled, -1] = 0
            self.sampled= np.random.choice(np.arange(self.n_users),size=int(self.K*len(self.user_ids)),replace=False)
            self.current_G[-1, self.sampled]=1
            update_model_counts(self.model_counts,self.sampled,np.full(len(self.sampled),-1))

class FixedHierarchicalFederatedLearningTopology(BaseTopology):

    """
    Splits users into subgroups, with a server sampling each of them.
    """

    def __init__(self, n_sub_servers=3, K=0.1):

        self.n_sub_servers=3
        self.K=K

    def create_subgroups(self, n_users,n_sub_servers):
        users = np.random.permutation(np.arange(n_users))
        k = int(n_users / self.n_sub_servers)
        self.subgroups={}
        idx=0
        for i in range(n_sub_servers):
            _users = users[i*k:min(len(users),k*(i+1))]
            self.subgroups[i] = _users

    def _fit(self, dataset):
        
        super()._fit(dataset, 1+self.n_sub_servers)
        self.user_ids = np.arange(dataset.n_users)
        self.is_aggregator[-(1+self.n_sub_servers):] = 1
        self.has_model[-1] = 1
        self.create_subgroups(dataset.n_users, self.n_sub_servers)
        self.subservers = np.arange(self.n_sub_servers) + dataset.n_users
        self.current_G[-1,self.subservers]=1
        self.current_G[self.subservers, -1] = 1
        self.two_way_node = np.ones(self.current_G.shape[0]).astype(int_dtype)
        self.two_way_node[self.subservers]=0
        self.step=3
        self.valid_attack_nodes = np.zeros(self.current_G.shape[0])
        self.valid_attack_nodes[self.subservers] = 1
        self._update()
        
        self.has_data[:self.n_users] = 1
        self.seeds = [self.current_G.shape[0]-1] 
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes[dataset.n_users:] = 0


    def update(self,sim):

        self._update()

    def _update(self):
        
        #resample whenever the central server receives models.
        self.step = ((self.step+1) % 4)

        if self.step==0:

            self.current_G.fill(0)
            self.current_G[-1, self.subservers] = 1
    
            update_model_counts(self.model_counts, self.subservers,np.array([self.current_G.shape[0]-1]))
        
        elif self.step==1:
            self.current_G[-1, self.subservers] = 0
            self.sampled_subgroups = {
                i:np.random.choice(self.subgroups[i], size=int(len(self.subgroups[i]*self.K)),replace=False)
                for i in self.subgroups
            }
            for i, s in enumerate(self.subservers):
                self.current_G[s, self.sampled_subgroups[i]] = 1

            update_model_counts(self.model_counts,np.concatenate([
                self.sampled_subgroups[i] for i in self.sampled_subgroups
            ]), self.subservers)

        elif self.step==2:

            for i, s in enumerate(self.subservers):
                self.current_G[s, self.sampled_subgroups[i]] = 0
                self.current_G[self.sampled_subgroups[i],s] = 1

            update_model_counts(self.model_counts,
            np.concatenate([
                [s]*len(self.sampled_subgroups[i]) for s, i in zip(self.subservers, self.sampled_subgroups) 
            ]),np.concatenate([
                self.sampled_subgroups[i] for i in self.sampled_subgroups
            ]) )

        elif self.step==3:
            for i, s in enumerate(self.subservers):
                
                self.current_G[self.sampled_subgroups[i],s] = 0
                self.current_G[self.subservers,-1] = 1
                update_model_counts(self.model_counts,
                                np.array([self.current_G.shape[0]-1] * len(self.subservers)),
                                self.subservers)
                

            


        
        

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
        self.has_model[-1] = 1
        self.has_data[:self.n_users] = 1
        self.seeds = [self.n_users] 
        self.two_way_node = np.zeros(self.has_data.shape).astype(int_dtype)
        self.two_way_node[-1] = 1
        self.valid_attack_nodes = np.ones(self.current_G.shape[0])
        self.valid_attack_nodes[-1] = 1
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes[dataset.n_users:] = 0
        self.sample()

    def sample(self):

        self.samples_array = np.random.randint(0, len(self.user_ids),[self.n_walks, self.walk_length+2])
        
        self.samples_array[:,0] = len(self.user_ids)
        self.samples_array[:,-1] = len(self.user_ids)
        self.current_G[self.samples_array[:,0], self.samples_array[:,1]] = 1
        self.walk_step=0
        update_model_counts(self.model_counts,self.samples_array[:,1],self.samples_array[:,0])
        
        
    def _update(self, sim):

        self.walk_step+=1 
        if self.walk_step>self.samples_array.shape[1]-2:
            self.current_G[self.samples_array[:,self.walk_step-1],self.samples_array[:,self.walk_step]] = 0
            self.sample()
        else:
            self.current_G[self.samples_array[:,self.walk_step-1],self.samples_array[:,self.walk_step]] = 0
            
            self.current_G[self.samples_array[:,self.walk_step],self.samples_array[:,self.walk_step+1]] = 1
            update_model_counts(self.model_counts,self.samples_array[:,self.walk_step+1],self.samples_array[:,self.walk_step])



class RandomGraphTopology(BaseTopology):

    def __init__(self, density=0.01, n_aggregators=0):

        self.density=density
        self.n_aggregators=n_aggregators
        super( RandomGraphTopology, self).__init__()
        pass

    def _fit(self, dataset):
        
        super()._fit(dataset, self.n_aggregators)
        self.user_ids = np.arange(dataset.n_users)
        self.is_aggregator[-self.n_aggregators:] = 1
        self.has_model[self.user_ids] = 1
        self.has_data[:self.n_users] = 1
        self.seeds = np.arange(dataset.n_users)
        self.two_way_node = np.zeros(self.has_data.shape).astype(int_dtype)
        self.two_way_node[-self.n_aggregators:] = 1
        mask = np.random.random(self.current_G.shape) < self.density
        self.current_G[mask>0] = 1
        self.valid_attack_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes[dataset.n_users:] = 0

    def _update(self,sim):
        return



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
        self.sampled = np.random.randint(0, len(self.user_ids), [int(len(self.user_ids)*self.density),2])
        self.current_G[self.sampled[:,0],self.sampled[:,1]] = 1
        self.seeds = self.sampled[:,0]
        self.has_model[self.sampled[:,0]] = 1
        self.valid_attack_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])
    
    def _update(self,sim):
        
        self.current_G[self.sampled[:,0],self.sampled[:,1]] = 0
        self.sampled[:,0] = self.sampled[:,1].copy()
        self.sampled[:,1] = np.random.randint(0, len(self.user_ids), [int(len(self.user_ids)*self.density)])
        self.current_G[self.sampled[:,0],self.sampled[:,1]] = 1
        update_model_counts(
            self.model_counts,
            self.sampled[:,1],
            self.sampled[:,0]
        )


class MultiServerRandomWalk(BaseTopology):


    def init(self, density=0.1, n_servers=1, server_prob=0.1):

        self.density = density
        self.n_servers = n_servers
        self.server_prob=server_prob

     def _fit(self, dataset):
        
        super()._fit(dataset, self.n_servers)
        self.user_ids = np.arange(dataset.n_users)
        self.is_aggregator[-self.n_aggregators:] = 1
        self.has_data[:self.n_users] = 1
        self.two_way_node = np.zeros(self.has_data.shape).astype(int_dtype)
        self.two_way_node[-self.n_aggregators:] = 1
        mask = np.random.random(self.current_G.shape) < self.density
        self.sampled = np.random.randint(0, len(self.user_ids), [int(len(self.user_ids)*self.density),2])
        self.current_G[self.sampled[:,0],self.sampled[:,1]] = 1
        self.seeds = self.sampled[:,0]
        self.has_model[self.sampled[:,0]] = 1
        self.valid_attack_nodes = np.ones(self.current_G.shape[0])
        self.valid_victim_nodes = np.ones(self.current_G.shape[0])

    






            

