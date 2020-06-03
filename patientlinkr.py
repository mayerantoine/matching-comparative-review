import sys
import logging
import numpy
import pandas

import recordlinkage
from recordlinkage.index import Block
from recordlinkage.base import BaseIndexAlgorithm
from recordlinkage.index import Block,SortedNeighbourhood
from recordlinkage.compare import Exact, String, Numeric, Geographic, Date
from recordlinkage.preprocessing import phonetic
from recordlinkage.compare import Exact, String, Numeric, Geographic, Date
from recordlinkage.preprocessing import clean

from string import ascii_lowercase
from metaphone import doublemetaphone
from sklearn.model_selection import KFold,StratifiedKFold
import uuid 
import copy
from sklearn.base import clone
from functools import reduce
import warnings
warnings.filterwarnings("ignore")



## add special dunder repr in the class
## add special dunder str in the class
## false positive rate
## cross_val_predict
## cross_val_score
### roc_curve 
## get FP, TP,FN, TN
## use hasattr function to check attribute
## can i execute a steps without fitting 

class Comparator():
    
    def __init__(self,compare_on = None, **kwargs):
        super(Comparator,self).__init__(**kwargs)
        self.compare_on = compare_on
        self.comparator = recordlinkage.Compare()
            
    def compute_compare(self,pairs,df_a,df_b=None):
        for comparison in self.compare_on:
            vartype,field,method, threshold, label = self._unpack_dict(**comparison)
            if vartype == 'string':
                self.comparator.add(String(field,field,method,threshold,0,label))
            if vartype == 'exact':
                self.comparator.add(Exact(field,field,1,0,0,label))
                
        return  self.comparator.compute(pairs,df_a,df_b)
    
    def _unpack_dict(self,vartype = None,field= None,method = None,threshold = None, code= None):
        return vartype,field,method, threshold, code


class BlockUnion(BaseIndexAlgorithm):
    def __init__(self,block_on = None, **kwargs):
        
        super(BlockUnion,self).__init__(**kwargs)
        self.block_on = block_on         
         
    def _link_index(self, df_a, df_b):
        indexer = recordlinkage.Index()
          
        for blocking_keys in self.block_on:
            indexer.add(Block(blocking_keys))
        
        return  indexer.index(df_a,df_b)
     
    def _dedup_index(self,df_a):
         
        indexer = recordlinkage.Index()
         
        for blocking_keys in self.block_on:
            indexer.add(Block(blocking_keys))
          
        return  indexer.index(df_a)
          
                 
class ExactMatchingClassifier():
    """    Wrapper class to emulate exact deterministic matching     """
          
    def __init__(self, **kwargs):
          super(ExactMatchingClassifier,self).__init__(**kwargs)
               
    def fit_predict(self,candidate_pairs):
          return candidate_pairs


class WeightedAverageClassifier():
    
    def __init__(self,threshold,weight_factor,**kwargs):
        super(WeightedAverageClassifier,self).__init__(**kwargs)
        self.threshold = threshold
        self.weight_factor = weight_factor
    
    def fit_predict(self,comparison_vectors):
        df_score = comparison_vectors.copy()
        weighted_list =[]
        factor_sum = 0
        for field,wf in self.weight_factor.items():
            weighted_list.append(df_score[field]*int(wf))
            factor_sum += wf
        weighted_sum = reduce(lambda x, y: x.add(y, fill_value=0), weighted_list)
        df_score['score'] = weighted_sum/factor_sum
        matches = df_score[df_score['score'] >=self.threshold]
    
        return matches.index           
 
    
class SimSumClassifier():
     
    def __init__(self, threshold,weights = None,**kwargs):
        super(SimSumClassifier,self).__init__(**kwargs)
        self.threshold = threshold
        self.weights = weights       
          
    def fit_predict(self,comparison_vectors):
        
        comparison_vectors['score'] = comparison_vectors.sum(axis =1)
        #features.reset_index(inplace = True)
        match = comparison_vectors[comparison_vectors['score'] >= self.threshold ]
         
        return match.index


       
class Strategy():
    """ The purpose of the pipeline is to assemble several steps that 
    can be  cross-validated together while setting different parameters."""
    
    def __init__(self,steps):
        
        # validate steps 
        
        self.indexer = steps.get("indexer")
        self.comparator = steps.get("comparator")
        self.classifier = steps.get("classifier")
        self.pairs = None
        self.features = None
        self.links_pred = None
    
    
    def _extract_features(self,df_a,df_b=None):
        self.pairs = self.indexer.index(df_a,df_b)
        return self.comparator.compute_compare(self.pairs,df_a,df_b)
      
            
    def fit(self,df_a,links_true,train_size,df_b=None):
      
        self.features = self._extract_features(df_a,df_b)
        golden_match_index = self.features.index & links_true.index
        train_index = int(len(self.features)*train_size)
        self.classifier.fit(self.features[:train_index],golden_match_index)
    
    
    def predict(self,df_a,df_b=None):
        
        if len(self.features) >0:
            self.links_pred = self.classifier.predict(self.features)
        else:    
            self.features = self._extract_features(df_a,df_b)
            self.links_pred = self.classifier.predict(self.features)
        
        return self.links_pred
      
      
    def fit_predict(self,df_a,df_b=None):
        self.features = self._extract_features(df_a,df_b)
        self.links_pred = self.classifier.fit_predict(self.features)
        
        return self.links_pred
    

def cross_val_score(classifier,comparison_vector,link_true,cv = 5, method = 'fscore'):
    skfolds = StratifiedKFold(n_splits = cv)
    
    y = pandas.Series(0, index=comparison_vector.index)
    y.loc[link_true.index & comparison_vector.index] = 1
    
    X_train = comparison_vector.values
    y_train = y.values
    scores = []
    for train_index, test_index in skfolds.split(X_train,y_train):
        #clone_clf = clone(classifier)
        classifier_copy = copy.deepcopy(classifier)
        X_train_folds = comparison_vector.iloc[train_index]  #X_train[train_index]
        X_test_folds  = comparison_vector.iloc[test_index]  #X_train[test_index]
        y_train_folds = X_train_folds.index &  link_true.index #y_train[train_index]
        y_test_folds = X_test_folds.index & link_true.index

        # Train the classifier
        #print(y_train_folds.shape)
        classifier_copy.fit(X_train_folds, y_train_folds)

        # predict matches for the test
        #print(X_test_folds)
        y_pred = classifier_copy.predict(X_test_folds)
        
        if(method == 'fscore'):
            score = recordlinkage.fscore(y_test_folds,y_pred)
        elif(method == 'precision'):
            score = recordlinkage.precision(y_test_folds,y_pred)
        elif(method == 'recall'):
            score = recordlinkage.recall(y_test_folds,y_pred)
        elif(method == 'accuracy'):
            score = recordlinkage.accuracy(y_test_folds,y_pred,len(comparison_vector))
        elif(method == 'specificity'):
            score = recordlinkage.specificity(y_test_folds,y_pred,len(comparison_vector))
        scores.append(score)

    scores = numpy.array(scores)
    return scores
        
    
def metrics(links_true,links_pred,comparison_vector):

    if len(links_pred) > 0 :
        # confusion matrix
        matrix  = recordlinkage.confusion_matrix(links_true, links_pred, len(comparison_vector))
        
        # precision
        precision  = recordlinkage.precision(links_true, links_pred)

        # precision
        recall  = recordlinkage.recall(links_true, links_pred)

        # The F-score for this classification is
        fscore = recordlinkage.fscore(links_true, links_pred)

        return matrix, precision, recall,fscore
    else :
        return 0, 0, 0, 0      


def cross_val_predict(classifier,comparison_vector,link_true,cv = 5 , method ='predict'):
        skfolds = StratifiedKFold(n_splits = cv)
        
        y = pandas.Series(0, index=comparison_vector.index)
        y.loc[link_true.index & comparison_vector.index] = 1
        
        X_train = comparison_vector.values
        y_train = y.values
        
        results = pandas.DataFrame()
        for train_index, test_index in skfolds.split(X_train,y_train):
            #clone_clf = clone(classifier)
            classifier_copy = copy.deepcopy(classifier)
            X_train_folds = comparison_vector.iloc[train_index]  #X_train[train_index]
            X_test_folds  = comparison_vector.iloc[test_index]  #X_train[test_index]
            y_train_folds = X_train_folds.index &  link_true.index #y_train[train_index]
            y_test_folds = X_test_folds.index & link_true.index

            # Train the classifier
            #print(y_train_folds.shape)
            classifier_copy.fit(X_train_folds, y_train_folds)

            # predict matches for the test
            #print(X_test_folds)
            
            if(method == 'predict'):
                y_pred = classifier_copy.predict(X_test_folds)
                results = pandas.concat([results,y_pred.to_frame()])
            elif(method == 'predict_proba'):
                predict_proba = pandas.DataFrame(classifier.prob(X_test_folds))
                predict_proba.columns = ['score']
                results = pandas.concat([results,predict_proba])
            elif(method == 'decision_function'):
                decision_match = classifier.kernel.decision_function(comparison_vector.values)
                decision = pandas.Series(decision_match,index = comparison_vector.index)
                df_decision = pandas.DataFrame(decision)
                results = pandas.concat([results,df_decision])

        return pandas.MultiIndex.from_frame(results)
   
        
class DisjointSet(object):

    def __init__(self):
        self.leader = {} # maps a member to the group's leader
        self.group = {} # maps a group leader to the group (which is a set)

    def add(self, a, b):
        leadera = self.leader.get(a)
        leaderb = self.leader.get(b)
        if leadera is not None:
            if leaderb is not None:
                if leadera == leaderb: return # nothing to do
                groupa = self.group[leadera]
                groupb = self.group[leaderb]
                if len(groupa) < len(groupb):
                    a, leadera, groupa, b, leaderb, groupb = b, leaderb, groupb, a, leadera, groupa
                groupa |= groupb
                del self.group[leaderb]
                for k in groupb:
                    self.leader[k] = leadera
            else:
                self.group[leadera].add(b)
                self.leader[b] = leadera
        else:
            if leaderb is not None:
                self.group[leaderb].add(a)
                self.leader[a] = leaderb
            else:
                self.leader[a] = self.leader[b] = a
                self.group[a] = set([a, b])
  
      
def _create_pseudo_id(df_a):
    
    #first letter of given name
    given_name_1 = pandas.Series(clean(df_a['given_name']).str[:1].dropna().apply(lambda x: alphabet_position(x)), name = 'given_name_1')
    # second letter of the given name
    given_name_2 = pandas.Series(clean(df_a['given_name']).str[1:2].dropna().apply(lambda x:alphabet_position(x)),name = 'given_name_2')
    # first letter of the surname
    surname_1 = pandas.Series(clean(df_a['surname']).str[:1].dropna().apply(lambda x: alphabet_position(x)), name = 'surname_1')
    # second letter of the surname
    surname_2 = pandas.Series(clean(df_a['surname']).str[1:2].dropna().apply(lambda x:alphabet_position(x)),name = 'surname_2')
    
    #sex 
    surname_2 = pandas.Series(clean(df_a['surname']).str[1:2].dropna().apply(lambda x:alphabet_position(x)),name = 'surname_2')
    # year of birth
    yearb = pandas.Series(df_a['YearB'].dropna().astype(str).replace('nan',''), name ='YearB')
    # day of birth
    dayb = pandas.Series(df_a['DayB'].dropna().astype(str).replace('nan',''),name = 'DayB')
    df_u = pandas.concat([given_name_1,given_name_2,surname_1,surname_2,yearb,dayb], axis=1,sort=False)
    df_u = df_u.fillna('')

    return df_u.apply(lambda x: ''.join(x), axis=1)

   
def get_unique(df_a,df_true_links):
    
    ds = DisjointSet()
    df_true_links_frame = pandas.DataFrame()
    if(isinstance(df_true_links,pandas.MultiIndex)):
        df_true_links_frame = df_true_links.to_frame(index=False)
        df_true_links_frame.columns=['rec_id_1','rec_id_2']
    else:
        df_true_links_frame = df_true_links
        df_true_links_frame.columns=['rec_id_1','rec_id_2']  
    for index, row in df_true_links_frame.iterrows():
        ds.add(row['rec_id_1'],row['rec_id_2'])
    link_uid = []
    for el, item in ds.group.items():
        id = uuid.uuid4() 
        for val in item:
            link_uid.append((str(id),val))
    df_link_uid = pandas.MultiIndex.from_tuples(link_uid,names=('uuid', 'rec_id')).to_frame(index=False).set_index('rec_id')
    df_a_new = df_a.merge(df_link_uid,how='left',left_on='rec_id', right_on='rec_id')
    df_a_new.loc[df_a_new['uuid'].isnull(),'uuid'] = df_a_new[df_a_new['uuid'].isnull()]['uuid'].apply(lambda x:str(uuid.uuid4()))
    return df_a_new['uuid'].nunique()


def alphabet_position(char):
    LETTERS = {letter: str(index) for index, letter in enumerate(ascii_lowercase, start=1)} 
    char.replace(" ", "")
    if (char is 'nan') or (not char) or (char is ' ') or (char.isnumeric()):
        return ''
    else:
        return str(LETTERS[char])
    

