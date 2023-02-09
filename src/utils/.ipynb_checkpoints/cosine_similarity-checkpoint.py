from utils.feature_extractor import *
from torch import nn
from torch.utils.data import DataLoader
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import math



def compute_similarity(a, b) -> float:
    d = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1/(1 + math.exp(d))


def get_similarity_vec(avgs):
    l = len(avgs) - 1

    d = lambda vi, vj : 1 - (np.dot(vi, vj)/(np.linalg.norm(vi) * np.linalg.norm(vj)))
    s = lambda vi, vj : 1 / (1 + math.exp(d(vi,vj)))
    
    sims = []

    for i in range(l):
        sims.append(s(avgs[l], avgs[i]))
    
    # in the paper, they normalize over the similarity scores for the rest of the classes
    # however I don't think this is necessary since it looks like they're just doing this
    # for the sake of the visualization. If we encounter problems later, it's worth taking
    # a look at this again. However rn, it messes it up, and says "t-shirt" is most similar
    # to ankle boot, whereas if we don't normalize, we get sneaker=boot which is right

    # for i in range(l):
    #    sims[i] = sims[i] / sum(sims, i+1)
        
    return sims

    

def get_lda_avgs(X, y, subset_size):
    trans_act = LinearDiscriminantAnalysis().fit_transform(X,y)
    
    # group the data by classes and get avg class activation
    class_splits_trans = []

    for i in range(1,int(len(trans_act)/subset_size) + 1):
        idx = subset_size * i
        class_splits_trans.append(trans_act[idx-subset_size:idx])
    
    avgs = np.mean(class_splits_trans, axis=1, dtype=np.float64)
    
    return avgs
    

def extract_features(model: nn.Module, classes, class_subsets, subset_size):
    X = []
    y = []

    for i in range(len(classes)):
        for img, c in class_subsets[i]:
            with torch.no_grad():
                feature = model(img)
            X.append(feature['input_layer'].numpy().flatten())
            y.append(classes[i])
    
    return np.array(X), np.array(y), subset_size
    

def generate_dls(dl : DataLoader, classes: []): 
    class_subsets = []
    classes_idxs = []

    for class_idx in classes:
        # Struggling to work with our subsets, this works faster
        classes_idx = np.where((np.array(dl.targets) == class_idx))[0]
        class_subset = torch.utils.data.Subset(dl, classes_idx)
        class_subsets.append(class_subset)
        classes_idxs.append(classes_idx)
        
    subset_size = len(classes_idx)
    
    return class_subsets, classes_idxs, subset_size
