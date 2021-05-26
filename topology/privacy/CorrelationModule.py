import numpy as np 
from dcor import distance_correlation

class CorrelationModule():

    def __init__(self,verbose=False, attack_percent=1.0):

        self.verbose=verbose
        self.attack_percent = attack_percent

    def fit(self, sim):

        self.sim = sim
        self.history=[]
    def evaluate(self):

        array = self.sim.topology.samples_array
        step = self.sim.topology.walk_step
        if step == 0:
            self.history.append(0)
            return 0
        Q = self.sim.Q1
        current = array[:,step]
        prev = array[:, step-1]
        corrs=[]
        for _idx in range(current.shape[0]):
            target = prev[_idx]
            if self.sim.has_data[target] > 0 and np.random.random() < self.attack_percent:
             
                Q_cur = Q[current[_idx]]
                Q_prev = Q[target]
                X = Q_prev - Q_cur
                Y = self.sim.dataset.train_rating_matrix[target]
                corr = distance_correlation(X, Y) 
                corrs.append(corr)
        if len(corrs) > 0:
            
            corr = np.mean(corrs)
            self.history.append(corr)
            if self.verbose:
                print("Distance correlation:", corr)
            return corr
        else:
            #print("nothing to evaluate")
            self.history.append(0)
            return 0