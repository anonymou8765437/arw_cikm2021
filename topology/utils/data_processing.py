import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from topology.properties import *
from sklearn.utils import shuffle
#from fdmf.utils.Graph import *

def get_rating_matrix(df, n_users, n_items, fill_value=0):

    X = np.full([n_users, n_items], fill_value).astype(np.float64)
    X[df['user_id'], df['item_id']] = df['rating']
    return X


def label_encode_train_test(train_df, test_df, return_encoders=False):

    X = pd.concat([train_df, test_df], axis=0)
    n_users = X['user_id'].nunique()
    n_items = X['item_id'].nunique()
    user_encoder = LabelEncoder().fit(X['user_id'])
    item_encoder = LabelEncoder().fit(X['item_id'])

    for df in [train_df, test_df]:

        for encoder, key in zip([user_encoder, item_encoder], ['user_id', 'item_id']):

            df[key] = encoder.transform(df[key])

    if not return_encoders:
        return train_df, test_df, n_users, n_items,
    else:
        return train_df, test_df, n_users, n_items, user_encoder, item_encoder

def stratified_train_test_split(df,
                                split_ratio=0.8,
                                min_ratings=0):

 
    test = []
    train = []

    for key, X in df.groupby('user_id'):

        # discard users with less than min ratings
        if X.shape[0] < min_ratings:

            continue

        msk = np.random.random(len(X)) < split_ratio

        if len(X[msk]) < 1:
            train.append(X)
            continue

        test.append(X[~msk])
        train.append(X[msk])

    return pd.DataFrame(pd.concat(train)), pd.DataFrame(pd.concat(test))


def leave_one_out_train_test_split(df, min_ratings=2):

    import numpy as np
    from sklearn.utils import shuffle
    test = []
    train = []
    min_ratings = min_ratings if min_ratings else 2

    for key, X in df.groupby('user_id'):

        # discard users with less than min ratings
        if X.shape[0] < min_ratings or X.shape[0] < 2:

            continue

        X = shuffle(X)
        test.append(X[-1:])
        train.append(X[:-1])

        tr_it = set(list(X[:-1]['item_id'].unique()))
        te_it = set(list(X[-1:]['item_id'].unique()))
        try:
            assert len(X[-1:]) == 1
            assert len(tr_it.intersection(te_it)) == 0
        except:
            print(tr_it, te_it)
    return pd.DataFrame(pd.concat(train)), pd.DataFrame(pd.concat(test))


class Dataset():

    def __init__(self, train_df, test_df,dset_type=None):
        
        if dset_type is None:
            self.train_df, self.test_df, self.n_users, self.n_items, self.user_encoder, self.item_encoder = label_encode_train_test(
                train_df, test_df, return_encoders=True)
            self.train_rating_matrix = get_rating_matrix(
                self.train_df, self.n_users, self.n_items)
            self.test_rating_matrix = get_rating_matrix(
                self.test_df, self.n_users, self.n_items)
        

class DatasetWithGraph():

    def __init__(self, train_df, test_df,dset_type=None,graph_loader=None, graph_type=None, graph_params=None):
        
        if dset_type is None:
            self.train_df, self.test_df, self.n_users, self.n_items = label_encode_train_test(
                train_df, test_df)
            self.train_rating_matrix = get_rating_matrix(
                self.train_df, self.n_users, self.n_items)
            self.test_rating_matrix = get_rating_matrix(
                self.test_df, self.n_users, self.n_items)


        if graph_type is None:

            self.G = RandomGraph(self.train_rating_matrix.shape[0], graph_params.density)


        


        

class FeatureDataset():

    def __init__(self, train_df, test_df, user_features, item_features):
        train_df['user_id']-=1
        test_df['user_id']-=1
        train_df['item_id']-=1
        test_df['item_id']-=1
        print(train_df['user_id'].max(), user_features.shape)

        def func(x):
            return [list(i) for i in x]
        
        
        self.user_features = user_features
        self.item_features = item_features

        self.train_df, self.test_df, self.n_users, self.n_items, user_encoder, item_encoder = label_encode_train_test(
            train_df, test_df, return_encoders=True)
        self.train_rating_matrix = get_rating_matrix(
            self.train_df, self.n_users, self.n_items)
        self.test_rating_matrix = get_rating_matrix(
            self.test_df, self.n_users, self.n_items)

        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

        return
        train_reverse_user = user_encoder.inverse_transform(train_df['user_id'].values)
        
        train_reverse_item = item_encoder.inverse_transform(train_df['item_id'].values)

        test_reverse_user = user_encoder.inverse_transform(test_df['user_id'].values)
        
        test_reverse_item = item_encoder.inverse_transform(test_df['item_id'].values)

        
        def process_features(feat_matrix):

            max_features=np.max(np.sum(feat_matrix, axis=1)).astype(np.int32)
            print(max_features)
            print(feat_matrix.shape)
            out=np.zeros([feat_matrix.shape[0], max_features])
            for i in range(feat_matrix.shape[0]):
                q=np.argwhere(feat_matrix[i]!=0).flatten()
                out[i, :len(q)]=q

            return out
            
        
        self.train_samples = np.concatenate([
                            train_df['user_id'].values.reshape(-1,1),
                            train_df['item_id'].values.reshape(-1,1),
                            user_features[train_reverse_user],
                            item_features[train_reverse_item],
                            ], axis=1)
        
        self.test_samples = np.concatenate([
                            test_df['user_id'].values.reshape(-1,1),
                            test_df['item_id'].values.reshape(-1,1),
                            user_features[test_reverse_user],
                            item_features[test_reverse_item],
                            ], axis=1)

        self.n_features = self.train_samples.shape[1]
        self.train_samples = process_features(self.train_samples)
        self.test_samples = process_features(self.test_samples)

        self.n_samples_train = np.sum(self.train_samples > 0, axis=1).astype(NUMPY_INT_DTYPE)
        self.n_samples_test = np.sum(self.test_samples > 0, axis=1).astype(NUMPY_INT_DTYPE)
        self.train_values = np.ones(self.train_samples.shape).astype(NUMPY_FLOAT_DTYPE)
        self.test_values = np.ones(self.test_samples.shape).astype(NUMPY_FLOAT_DTYPE)

def load_dataset(df, split_ratio=0.8, min_ratings=0, split_type='stratified',dset_type=None, graph_type=None, graph_params=None):

    if split_type == 'stratified':
        train_df, test_df = stratified_train_test_split(
            df, split_ratio=split_ratio, min_ratings=min_ratings)

    elif split_type == 'leave_one_out':

        train_df, test_df = leave_one_out_train_test_split(
            df, min_ratings=min_ratings)

    else:
        raise ValueError('Not implemented')
    

    return Dataset(train_df, test_df)
    # if graph_type is None and graph_params is None:

    #     return Dataset(train_df, test_df, dset_type=dset_type)

    # else:

    if graph_params is None:
        graph_params = GraphParams()

    return DatasetWithGraph(train_df, test_df, dset_type=dset_type, graph_type=graph_type, graph_params=graph_params)


def load_feature_dataset(df,user_features, item_features, split_ratio=0.8,min_ratings=0, split_type='stratified'):

    
    if split_type == 'stratified':
        train_df, test_df = stratified_train_test_split(
            df, split_ratio=split_ratio, min_ratings=min_ratings)

    elif split_type == 'leave_one_out':

        train_df, test_df = leave_one_out_train_test_split(
            df, min_ratings=min_ratings)

    else:
        raise ValueError('Not implemented')

    return FeatureDataset(train_df,test_df, user_features, item_features)


def load_ml100k(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):

    if not path:

        from topology.properties import get_project_root
        path = "{}/data/ml100k/u.data".format(get_project_root())

    df = pd.read_csv(
        path, names=['user_id', 'item_id', 'rating', 'dt'], sep='\t')
    df.drop('dt', 1)

    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)

def load_ml10m(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):

    if not path:

        from topology.properties import get_project_root
        path = "{}/data/ml-10M100K/ratings.dat".format(get_project_root())

    df = pd.read_csv(
        path, engine='python',names=['user_id', 'item_id', 'rating', 'dt'], sep='::')
    df.drop('dt', 1)

    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)

def load_tripadvisor(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):

    if not path:

        from topology.properties import get_project_root
        path = "{}/data/tripadvisor.csv".format(get_project_root())

    df = pd.read_csv(
        path)

    df=df.drop_duplicates(subset=['user_id', 'item_id'])
    #df.drop('dt', 1)

    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)


def load_ml100k_feature(split_ratio=0.8, min_ratings=0, split_type='stratified'):


    def process_features(feat_matrix):

        max_features=np.max(np.sum(feat_matrix, axis=1)).astype(np.int32)
        print(max_features)
        print(feat_matrix.shape)
        out=np.zeros([feat_matrix.shape[0], max_features])
        for i in range(feat_matrix.shape[0]):
            q=np.argwhere(feat_matrix[i]!=0).flatten()
            out[i, :len(q)]=q

        return out

    def get_item_features(directory='data/', feature_type='genre'):
        #we have very sparse features at this point
        #should contemplate maybe adding genre or something
        a = "movie id | movie title | release date | video release date | IMDb URL | unknown | Action |\
        Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy |\
        Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |Thriller | War | Western |"
        names = [i.strip() for i in a.split('|')][:-1]
        items = pd.read_csv(directory+'u.item', names=names, sep='|',encoding='latin1')
        items['movie id'] -= 1

        if feature_type == 'genre':
            genres= "unknown | Action |\
            Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy |\
            Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |Thriller | War | Western"
            genres = [i.strip() for i in genres.split('|')]
            features = np.zeros([len(items), len(genres)])
            for r in range(len(items)):
                row = items.loc[r]
                m_id = row['movie id']
                for g_idx, g in enumerate(genres):
                    if row[g] == 1:
                        features[m_id, g_idx] = 1
        
        return features

    def get_user_features(directory='data/'):
        #not nearly as sparse as the item feature vectors are.....
        users = pd.read_csv(directory+'u.user', names=['user_id','age','sex','job','zip'], sep='|')
        occupations = users['job'].unique()
        feas = [i for i in occupations] + ['M', 'F']
        fea = {i:idx for idx, i in enumerate(feas) }
        users['user_id'] -= 1
        features = np.zeros([len(users), len(fea)])
        for r in range(users.shape[0]):
            _r = users.iloc[r]
            j_idx = fea[_r['job']]
            s_idx = fea[_r['sex']]
            u_idx = _r['user_id']
            features[u_idx,j_idx]=1
            features[u_idx, s_idx]=1
        return features

    from fdmf.utils.meta import get_project_root
    path = "{}/data/ml100k/".format(get_project_root())

    df = pd.read_csv(
        path+'u.data', names=['user_id', 'item_id', 'rating', 'dt'], sep='\t')
    df.drop('dt', 1)
    user_features = get_user_features(path)
    item_features = get_item_features(path)
    return load_feature_dataset(df, user_features, item_features,
    split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)



def load_yelp(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):
    from topology.properties import get_project_root
    if not path:

        from topology.properties import get_project_root
        path = "{}/data/yelp.csv".format(get_project_root())

    df = pd.read_csv(
        path, names=['user_id', 'item_id', 'rating'], sep=',',skiprows=1)
    #print(df['rating'].unique())
    #input()
    #df.drop('dt', 1)
    
    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)




def load_yelp2(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):
    from topology.properties import get_project_root
    if not path:

        
        path = "{}/data/yelp2.csv".format(get_project_root())

    df = pd.read_csv(
        path, names=['index','user_id', 'item_id', 'rating'], sep=',',skiprows=1)
    #print(df['rating'].unique())
    #input()
    df.drop('index', 1)
    
    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)

def load_lastfm_artist(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):

    if not path:

        from fdmf.utils.meta import get_project_root
        path = "{}/data/lastfm-dataset-1K/ratings.dat".format(get_project_root())

    df = pd.read_csv(
        path)

    df=df.drop_duplicates(subset=['user_id','item_id'])
    df['rating'] = np.ones(len(df)).astype(np.float64)
    #print(df['rating'].unique())
    #input()
    #df.drop('dt', 1)
    
    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)




def load_yelp_librahu(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):

    if not path:

        from fdmf.utils.meta import get_project_root
        path = "{}/data/Yelp/user_business.dat".format(get_project_root())

    df = pd.read_csv(
        path, names=['user_id', 'item_id', 'rating'], sep='\t',skiprows=1)
    #print(df['rating'].unique())
    #input()
    #df.drop('dt', 1)
    
    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)


def load_yelp_restaurant(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):

    if not path:

        from fdmf.utils.meta import get_project_root
        path = "{}/data/yelp-restaurant/ratings.csv".format(get_project_root())

    df = pd.read_csv(path)
    #input()
    #df.drop('dt', 1)
    
    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type,dset_type='reviews')




def load_ml1m(path=None, split_ratio=0.8, min_ratings=0, split_type='stratified'):
    from topology.properties import get_project_root
    if not path:

      
        path = "{}/data/ml-1m/ratings.dat".format(get_project_root())

    df = pd.read_csv(
        path, names=['user_id', 'item_id', 'rating','dt'], sep=',',skiprows=1)
    #print(df['rating'].unique())
    #input()
    df.drop('dt', 1)
    
    return load_dataset(df, split_ratio=split_ratio, min_ratings=min_ratings, split_type=split_type)


def load_adressa(path="/home/elias/git1/lastpyrec/pyrec/notebooks/development/drl-paper"):

    import pickle
    import pandas as pd
    train=pd.read_pickle("{}/adressa_train_p.pickle".format(path))
    test=pd.read_pickle("{}/adressa_test_p.pickle".format(path))
    users_filtered=pd.read_pickle("{}/adressa_users_filtered_p.pickle".format(path))
    users_test=pd.read_pickle("{}/adressa_users_test_p.pickle".format(path))
    items=pd.read_pickle("{}/adressa_items_p.pickle".format(path))
    item_features=pd.read_pickle("{}/adressa_item_features_p.pickle".format(path))
    user_features=pd.read_pickle("{}/adressa_user_features_p.pickle".format(path))
    userlist=pd.read_pickle("{}/adressa_userlist_p.pickle".format(path))
    
    x = pd.concat([train, test])
    u_enc = LabelEncoder().fit(x['user_id'])
    i_enc = LabelEncoder().fit(x['item_id'])

    for df in train, test:

        for col, enc in zip( ['user_id','item_id'], [u_enc, i_enc]):

            df[col] = enc.transform(df[col])

    train = pd.DataFrame({
        'user_id':train['user_id'],
        'item_id':train['item_id'],
        'rating':train['rating']
    })
    
    
    test = pd.DataFrame({
        'user_id':test['user_id'],
        'item_id':test['item_id'],
        'rating':test['rating']
    })

    return Dataset(train,test)


def load_yelp_reviews():
    
    import os
    import sklearn
    import time
    from sklearn import decomposition
    from sklearn.decomposition import LatentDirichletAllocation
    from pdfm import Constants
    from pdfm.nlp import topic_ensemble_caller, tfidf_creator
    from pdfm.properties import Properties
    from pdfm.utils import matrix_utils
    from pdfm.utils.file_names import FileNames
    from sklearn.externals import joblib

    topic_model_corpus_folder = FileNames.NLP_FOLDER + 'corpus/'
    tfidf_vectorizer_file_path = FileNames.generate_file_name(
            'corpus_review_id', '', topic_model_corpus_folder)[:-1] + '_tfidf.pkl'
    tfidf_vectorizer = tfidf_creator.load_tfidf(tfidf_vectorizer_file_path)
    review_bow_path = FileNames.generate_file_name(
            'bow_files_review_id', '', FileNames.TEXT_FILES_FOLDER)[:-1] + '/'

    review_docs, doc_ids, classes = tfidf_creator.create_docs(review_bow_path, 0)
    document_term_matrix = tfidf_vectorizer.transform(review_docs)
    
    matrix = document_term_matrix.toarray()
    #return matrix
    #ratings = np.squeeze(matrix.flatten())
   
    t=np.argwhere(matrix!=100)
    users=[]
    items=[]
    ratings=[]
    for i in range(matrix.shape[0]):
        users.append(np.full([matrix.shape[1]], i))
        items.append(np.arange(matrix.shape[1]))
        ratings.append(matrix[i])
        #print(matrix[i].mean())
    users=np.concatenate(users, axis=0)
    items = np.concatenate(items, axis=0)
    ratings=np.concatenate(ratings, axis=0)
    #ratings = matrix[users, items]
    #print(ratings.shape, users.shape, items.shape)
    df = pd.DataFrame({'user_id':users, 'item_id':items, 'rating':ratings})
    
    return load_dataset(df,dset_type='reviews')
    
    
    
    




def create_iid_nodes(train_df, n_nodes=10):

    """

    Create N uniform partitions of the training data.

    """
    from sklearn.preprocessing import LabelEncoder
    nodes = {}
    
    train_df['node_indices'] = np.random.randint( 0, n_nodes, [len(train_df)])
    train_df['node_indices'] = LabelEncoder().fit_transform(train_df['node_indices'])

    for node, df in train_df.groupby('node_indices'):

        nodes[node] = df.copy()

    return nodes

def create_shared_nodes(train_df, share=0.1):

    n_users = train_df['user_id'].nunique()
    msk = np.random.random(len(train_df)) < share
    shared_data = train_df[msk]
    train_df = train_df[~msk]
    shared_data['indice'] = np.random.randint(0,n_users, [len(shared_data)])
    train_df['indice'] = train_df['user_id']
    X = pd.concat([shared_data, train_df], axis=0)
    nodes={}
    for node, df in X.groupby('indice'):

        nodes[node] = df.copy()

    return nodes



def create_single_rating_nodes(train_df, n_nodes=None):


    nodes = {}

    for rating in range(1, 6):

        nodes[rating-1] = train_df[train_df['rating']==rating].copy()

    # print(len(nodes))
    # input()
    return nodes



def create_pareto_nodes(train_df, shape=1.0, n_nodes=100):
    
    
    pick_probs = np.random.pareto(shape, [n_nodes])
    pick_probs = pick_probs / np.sum(pick_probs)
    labels_names = np.arange(n_nodes)
    df_labels = np.random.choice(labels_names, p=pick_probs, size=len(train_df))
    df_labels = LabelEncoder().fit_transform(df_labels)
    train_df['label'] = df_labels
    
    
    nodes={}
    for node_id, df in train_df.groupby('label'):
        
        nodes[node_id] = df
    
    #sizes = [nodes[node_id].shape[0] for node_id in nodes]
    #sns.distplot(sizes, hist=False)
    
    return nodes

def create_user_nodes(train_df):

    """

    Do something useful

    """

    nodes = {}
    for node, df in train_df.groupby('user_id'):

        nodes[node] = df.copy()
    
    return nodes 

def iid_copy_user_distribution(train_df):

    user_nodes = create_user_nodes(train_df)
    sizes=[len(user_nodes[node]) for node in user_nodes]
    s = [np.full([s],i) for i, s in enumerate(sizes)]
    s= np.concatenate(s)
    print(s.shape, np.sum(sizes), len(train_df))
    train_df = shuffle(train_df)
    train_df['label'] = s
    nodes={}
    for node_id, df in train_df.groupby('label'):
        nodes[node_id] = df.copy()

    return nodes


def create_item_nodes(train_df):

    nodes = {}
    for node, df in train_df.groupby('item_id'):

        nodes[node] = df.copy()

    return nodes


def emulate_user_distribution(train_df):

    u_nodes = create_user_nodes(train_df)
    sizes = [len(u_nodes[u]) for u in u_nodes]
    nodes = {}
    n=0
    idx=0
    train_df = shuffle(train_df)
    while idx < train_df.shape[0]:

        nodes[n] = train_df[idx:idx+sizes[n]].copy()
        
        idx+=sizes[n]
        
       
        n+=1
    return nodes

def get_normal_distribution(train_df):
    mean = len(train_df) / train_df['user_id'].nunique()
    sizes=np.random.normal(mean,mean/4, [942]).astype(np.int32)
    nodes={}
    train_df = shuffle(train_df)
    _idx=0
    for node, size in enumerate(sizes):


        nodes[node] = train_df[_idx:_idx+size].copy()
        _idx += size

    return nodes


def create_similar_user_nodes(train_df, train_rating_matrix, n_nodes):

    from sklearn.cluster import KMeans
    print(train_rating_matrix.shape)
    print("computing kmeans for {} nodes".format(n_nodes))
    kmeans = KMeans(n_clusters=n_nodes).fit(train_rating_matrix)
    print("Kmeans computed")
    labels=kmeans.labels_
    print(labels.shape)
    nodes = {}
    for label in np.unique(labels):

        user_ids = np.argwhere(labels==label).flatten()
        print(user_ids)
        df = train_df[train_df['user_id'].isin(user_ids)].copy()
        nodes[label] = df

    import seaborn as sns
    from matplotlib import pyplot as plt
    sns.distplot([len(nodes[node_id]) for node_id in nodes])
    plt.savefig('dist.png')
    return nodes




def create_multi_user_nodes(train_df, n_nodes):

    """

    Randomly (but unevenly) distribute users amongst nodes.

    """
    from sklearn.preprocessing import LabelEncoder

    users = train_df['user_id'].unique()
    labels = np.random.randint(0, n_nodes,len(users))
    labels = LabelEncoder().fit_transform(labels)
    nodes = {}
    for node in np.unique(labels).flatten():

        user_ids = np.argwhere(labels==node).flatten()
        data = []
        for _  in  range(len(user_ids)):

            user_id = user_ids[_]
        
            data.append(train_df[train_df['user_id']==user_id])

        nodes[node] = shuffle(pd.concat(data, axis=0))

    return nodes

def create_negative_training_samples_bpr(train_matrix, n_negatives=5):
    all_items = np.arange(train_matrix.shape[1])
    data = []
    for u in range(train_matrix.shape[0]):


        items = np.argwhere(train_matrix[u] > 0)
        _neg_items = np.setdiff1d(all_items, items)

        neg_items = np.random.choice(_neg_items,size=len(items)*n_negatives, replace=True)
        pos_items = np.tile(items, n_negatives)
        
        
        pos_neg_items = np.concatenate([np.full([len(pos_items) * n_negatives], u).reshape(-1,1),pos_items.reshape(-1,1), neg_items.reshape(-1,1)],axis=1)
        data.append(pos_neg_items)
    
    data = np.concatenate(data, axis=0)
    shuffled_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffled_indices]
    return pd.DataFrame({

        'user_id': data[:,0].astype(NUMPY_INT_DTYPE), 'item_id_i':data[:,1].astype(NUMPY_INT_DTYPE), 'item_id_j':data[:,2].astype(NUMPY_INT_DTYPE)

    })



def create_negative_fake_training_samples_bpr(train_matrix, n_negatives=5, n_fake=5):
    all_items = np.arange(train_matrix.shape[1])
    data = []
    for u in range(train_matrix.shape[0]):


        items = np.argwhere(train_matrix[u] > 0)
        _neg_items = np.setdiff1d(all_items, items)
        fake_items = np.random.choice(_neg_items, size=int(n_fake * (len(items)*n_negatives)), replace=True)
        #_not_fake_items = np.setdiff1d(_neg_items, fake__items)
        not_fake_items = np.random.choice(all_items, size=len(fake_items),replace=True)
        fake_items = np.concatenate([np.full([len(fake_items)], u).reshape(-1,1),fake_items.reshape(-1,1), not_fake_items.reshape(-1,1)],axis=1)

        neg_items = np.random.choice(_neg_items,size=len(items)*n_negatives, replace=True)
        pos_items = np.tile(items, n_negatives)
        
        pos_neg_items = np.concatenate([np.full([len(pos_items) * n_negatives], u).reshape(-1,1),pos_items.reshape(-1,1), neg_items.reshape(-1,1)],axis=1)
        pos_neg_items = np.concatenate([pos_neg_items, fake_items],axis=0)
        data.append(pos_neg_items)
    
    data = np.concatenate(data, axis=0)
    shuffled_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffled_indices]
    return pd.DataFrame({

        'user_id': data[:,0].astype(NUMPY_INT_DTYPE), 'item_id_i':data[:,1].astype(NUMPY_INT_DTYPE), 'item_id_j':data[:,2].astype(NUMPY_INT_DTYPE)

    })




def create_negative_training_samples_sgd(train_matrix, n_negatives=5):
    print(n_negatives)
    all_items = np.arange(train_matrix.shape[1])
    data = []
    for u in range(train_matrix.shape[0]):


        items = np.argwhere(train_matrix[u] > 0)
        _neg_items = np.setdiff1d(all_items, items)

        neg_items = np.random.choice(_neg_items,size=min(len(items)*n_negatives, len(_neg_items)), replace=False)
        pos_items = items 
        rating = np.concatenate([
            train_matrix[u, items].flatten(),
            np.full(len(neg_items), 0),


            ], axis=0

        )


        pos_neg_items = np.concatenate([np.full([len(pos_items) + len(neg_items)], u).reshape(-1,1),
                        np.concatenate([pos_items.reshape(-1,1), 
                        neg_items.reshape(-1,1)],axis=0), 
                        rating.reshape(-1,1)],axis=1)
        data.append(pos_neg_items)
    
    data = np.concatenate(data, axis=0)
    shuffled_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffled_indices]
    
    df= pd.DataFrame({

        'user_id': data[:,0].astype(NUMPY_INT_DTYPE), 'item_id':data[:,1].astype(NUMPY_INT_DTYPE), 'rating':data[:,2].astype(NUMPY_FLOAT_DTYPE)

    })

    return df




    
    



