import numpy as np
from sklearn.decomposition import PCA
from topology.properties import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
class PrivacyModule():


    def __init__(self, topN=10, attack_percent=1.0):
        
        """
        Implements basic privacy attack against communicating nodes
        Args:
        attack_percent: percentage of available nodes to attack at any one iteration.
        topN : topN by which attack success is measured.

        """
        self.attack_percent=attack_percent
        self.topN=topN
        

    def fit(self, sim):
        
        self.sim=sim
        self.pos_items ={
            u:df['item_id_i'].unique() 
            for u,df in sim.train_df.groupby('user_id')
        }
        self.attack_success = []

    def get_results(self):
        return self.attack_success

    def _attack_node(self, attacker, victim,sim ):

        victim_data = np.argwhere(sim.dataset.train_rating_matrix[victim]>0)
        Q = sim.Q1
        X = Q[victim] - Q[attacker]
        changed = np.argwhere(X.sum(axis=1)!=0).flatten()
        
        if changed.shape[0] > 0:

            pca = PCA().fit(X[changed])
            pc = -pca.components_[0]
            pred = pc.dot(X.T)
            if self.topN is None:
                topN = len(victim_data)
            else:
                topN = self.topN
            
            a = np.argsort(pred)[:topN]
            b = np.argsort(pred)[-topN:]
            a = len(np.intersect1d(victim_data, a)) / topN
            b = len(np.intersect1d(victim_data, b)) / topN
            
            return max(a,b)
        
        else:
   
            return 0

    def attack(self):

        sim = self.sim
        topology = sim.topology
        attack_nodes = np.argwhere((sim.topology.model_counts* sim.topology.valid_attack_nodes) > 0 ).flatten()
        potential_attacks = np.argwhere(sim.G.T[attack_nodes] > 0)
        if len(potential_attacks)==0:
            self.attack_success.append(0)
            return
        
        mask = np.random.random(len(potential_attacks)) < self.attack_percent
        c = potential_attacks[mask]
        #print(len(potential_attacks)) 
        if len(c) > 0: 
            potential_attacks = potential_attacks[mask]
        
        precision=np.zeros(len(potential_attacks))
        
        for idx,attack in enumerate(potential_attacks):
        
            attacker = attack_nodes[attack[0]]
            victim = attack[1]
          
            if sim.topology.valid_victim_nodes[victim]==0:
               continue
            else:
                q = self._attack_node(
                    attacker,
                    victim, 
                    sim
                )
              
                precision[idx] = q
        success = precision.mean()
        
        if sim.verbose:
            print("Attack success:", success)
        
        self.attack_success.append(success)

        

class RepeatPrivacyModule(PrivacyModule):


    def fit(self, sim):
        
        super().fit(sim)
        self.attack_matrix = np.zeros([sim.G.shape[0],sim.G.shape[0], sim.n_factors])
        self.n_attacks = np.zeros([sim.G.shape[0],sim.n_users])


    def _attack_node(self, attacker, victim,sim ):

        victim_data = np.argwhere(sim.dataset.train_rating_matrix[victim]>0)
        Q = sim.Q1
        X = Q[victim] - Q[attacker]
        changed = np.argwhere(X.sum(axis=1)!=0).flatten()
        
        if changed.shape[0] > 0:
            
            self.n_attacks[attacker,victim] += 1
            pca = PCA().fit(X[changed])
            pc = -pca.components_[0]
            self.attack_matrix[attacker,victim] += pc
            pred = (self.attack_matrix[attacker,victim]/self.n_attacks[attacker,victim]).dot(X.T)
            if self.topN is None:
                topN = len(victim_data)
            else:
                topN = self.topN
            
            a = np.argsort(pred)[:topN]
            b = np.argsort(pred)[-topN:]
            a = len(np.intersect1d(victim_data, a)) / topN
            b = len(np.intersect1d(victim_data, b)) / topN
            
            return max(a,b)
        
        else:
   
            return 0


class FeatureInferenceModule(PrivacyModule):

    """
    Only configured to work with ml100k.
    Further splits the dataset, removing a number of users to employ only for training the attack model

    """

    def __init__(self,verbose=True, model_type='logistic',topN=10, attack_percent=1.0, split_ratio=0.1, attack_attributes=['sex','occ']):

        self.model_type = model_type
        self.topN = topN
        self.verbose=verbose
        self.split_ratio = split_ratio
        self.attack_attributes=attack_attributes
        self.attack_percent = attack_percent

    def fit(self, sim):
        
        super().fit(sim)
        self.attack_matrix = np.zeros([sim.G.shape[0],sim.G.shape[0], sim.n_factors])
        self.n_attacks = np.zeros([sim.G.shape[0],sim.n_users])
        self.attack_success = {attr:[] for attr in self.attack_attributes}

    def pre_fit(self, sim):

        self.sim = sim 
        
        
       

        if self.model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.models = {attr:LogisticRegression(penalty='l1', C=8.0) for attr in self.attack_attributes}

        elif self.model_type=='MLP':
            from sklearn.neural_network import MLPClassifier
            self.models = {attr:MLPClassifier() for attr in self.attack_attributes}

        try:
            self._fit_model()

        except:

            raise ValueError('Module only configured to work with ML datasets')

    def _fit_model(self):
        import pandas as pd 
        path = "{}/data/ml100k/u.user".format(get_project_root())
        attr = pd.read_csv(path, header=None, sep='|',names=['user_id','age','sex','occ','zip'])
        attr['sex'] = attr['sex'].apply(lambda x : 1 if x=='M' else 0).astype(np.int32)
        occ_encoder = LabelEncoder().fit(attr['occ'])
        attr['occ'] = occ_encoder.transform(attr['occ'])
        attr['user_id'] = self.sim.dataset.user_encoder.transform(attr['user_id'])
        self.attr = attr
        #now split the dataset some more
        index = np.random.permutation(self.sim.dataset.n_users)
        train = index[:int(self.split_ratio * len(index))]
        test = index[int(self.split_ratio * len(index)):]
        train_encoder = LabelEncoder().fit(train)
        test_encoder = LabelEncoder().fit(test)
        tr = self.sim.dataset.train_rating_matrix
        te = self.sim.dataset.test_rating_matrix
        attr = attr.sort_values(by=['user_id'])
        train_data = tr[train] + te[train]
        attr_train = attr.loc[train]
        attr_test = attr.loc[test]
        self.attr_test = {attr:attr_test[attr].values for attr in self.attack_attributes}
        self.sim.dataset.train_rating_matrix = tr[test]
        self.sim.dataset.test_rating_matrix = te[test]
        self.sim.dataset.train_df = self.sim.dataset.train_df[self.sim.dataset.train_df['user_id'].isin(test)]
        self.sim.dataset.test_df = self.sim.dataset.test_df[self.sim.dataset.test_df['user_id'].isin(test)]
        new_index = {test[i]:i for i in range(len(test))}
        self.sim.dataset.train_df['user_id']=self.sim.dataset.train_df['user_id'].apply(lambda x:new_index[x])
        self.sim.dataset.test_df['user_id']=self.sim.dataset.test_df['user_id'].apply(lambda x:new_index[x])
        self.sim.dataset.n_users = self.sim.dataset.train_rating_matrix.shape[0]
        for attr in self.attack_attributes:

            self.models[attr].fit(train_data, attr_train[attr])


    
    def attack(self):

        sim = self.sim
        topology = sim.topology
        attack_nodes = np.argwhere((sim.topology.model_counts* sim.topology.valid_attack_nodes) > 0 ).flatten()
        potential_attacks = np.argwhere(sim.G.T[attack_nodes] > 0)
        if len(potential_attacks)==0:
            for attr in self.attack_attributes:
                self.attack_success[attr].append(0)
            return
        
        mask = np.random.random(len(potential_attacks)) < self.attack_percent
        c = potential_attacks[mask]
        #print(len(potential_attacks)) 
        if len(c) > 0: 
            potential_attacks = potential_attacks[mask]
        
        precision={attr:np.zeros(len(potential_attacks)) for attr in self.attack_attributes}
        
        for idx,attack in enumerate(potential_attacks):
        
            attacker = attack_nodes[attack[0]]
            victim = attack[1]
          
            if sim.topology.valid_victim_nodes[victim]==0:
               continue
            else:
                q = self._attack_node(
                    attacker,
                    victim, 
                    sim
                )

                for attr in self.attack_attributes:
                    precision[attr][idx] = q[attr]
        
        for attr in self.attack_attributes:

            success = precision[attr].mean()
            if self.verbose:
                print("Attack attribute '{}' success: {}".format(attr, success))

            self.attack_success[attr].append(success)


    def _attack_node(self, attacker, victim,sim ):

        victim_data = np.argwhere(sim.dataset.train_rating_matrix[victim]>0)
        Q = sim.Q1
        X = Q[victim] - Q[attacker]
        changed = np.argwhere(X.sum(axis=1)!=0).flatten()
        out={attr:0 for attr in self.attack_attributes}
        if changed.shape[0] > 0:
            
            self.n_attacks[attacker,victim] += 1
            pca = PCA().fit(X[changed])
            pc = -pca.components_[0]
            self.attack_matrix[attacker,victim] += pc
            pred = (self.attack_matrix[attacker,victim]/self.n_attacks[attacker,victim]).dot(X.T)
            if self.topN is None:
                topN = len(victim_data)
            else:
                topN = self.topN
            
            a = np.argsort(pred)[:topN]
            b = np.argsort(pred)[-topN:]
            aa = len(np.intersect1d(victim_data, a)) / topN
            bb = len(np.intersect1d(victim_data, b)) / topN
            
            if aa > bb:

                top_items = a
            
            else:
                
                top_items = b


            pred_vector = np.zeros([1, self.sim.dataset.n_items])
            pred_vector[0,top_items] = 1

            for attr in self.attack_attributes:
                
                model = self.models[attr]
                y_pred=model.predict(pred_vector)
                y_true = self.attr_test[attr][victim]
                out[attr] = 1 if y_true==y_pred[0] else 0

            return out
        
        else:
   
            return 0
        
            
            







