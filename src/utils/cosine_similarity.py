import math
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from torch import nn
from torch.utils.data import DataLoader

from utils.feature_extractor import *


def compute_similarity(a, b) -> float:
    d = 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1/(1 + math.exp(d))

def compute_similarity_mat(V):
    dotv = np.dot(V,V.T) #<vi | vj>
    normv_ = np.linalg.norm(V,2,axis = 1); normv_ = normv_[:,None] #||vi||
    normv = normv_*normv_.T #||vi|| ||vj||

    dv = 1-dotv/normv #1-<vi|vj>/||vi|| ||vj||
    s = 1/(1+np.exp(dv)) #1/(1+e^d)

    sm = s/np.sum(s,1) #normalize
    np.fill_diagonal(sm, math.nan) #remove diag
    
    return sm
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


def get_lda_avgs(X, y, subset_size=None):
    classes = np.unique(y).tolist()
    trans_act = LinearDiscriminantAnalysis().fit_transform(X,y)
    
    # group the data by classes and get avg class activation
    if subset_size is None:
        class_splits_trans = np.zeros([len(classes),len(trans_act[0])])
        flag = 0
        for cls in classes:
            idx = np.where(y == cls)[0]

            class_splits_trans[flag,:] = np.mean(trans_act[idx,:], axis=0)
            flag += 1
        
        return class_splits_trans
    else:
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


def get_avg_activations(model, dataset, classes, layer, device, batch_size=32):
    feature_extractor = FeatureExtractor(model, layers=layer).to(device)
    targets = np.array(dataset.targets)
    # Stores late layer activations
    X = torch.Tensor().to(device)
    # Stores activation labels
    y = None
   

    with torch.no_grad():
        for cls in classes:

            # Empty container to store batch activations
            class_activations = torch.Tensor().to(device)
            # Index array of class items
            idx = np.where(targets == cls)[0]
            #Create Dataloader with class data
            cls_data = torch.Tensor(dataset.data[idx]).to(device)
            cls_data_loader = DataLoader(
                cls_data, batch_size=batch_size)

            # Iterate over class data
            for batch in cls_data_loader:
               
                if len(batch.shape) > 3:
                    # Reshape batch for feature extractor
                    imgs = batch.permute(0, 3, 1, 2).to(device)
                else:
                    batch = batch.to(torch.float32)

                    imgs = batch.to(device)
                # Late layer activations for batch data
                batch_activations = feature_extractor(imgs)
                # Add batch activations to class activation tensor
                class_activations = torch.cat(
                    (class_activations, batch_activations[layer[0]]), dim=0)

                # Append batch's labels to the label array
                labels = np.full((len(batch)), cls)
                if y is not None:
                    y = np.concatenate((y, labels))
                else:
                    y = labels

            # Get average activation for entire class
            #avg_activations.append(np.mean(class_activations.numpy(), axis=0))
            X = torch.cat((X, torch.flatten(class_activations,1)), dim=0)
    # Return per class average activation avg from LDA space
    return get_lda_avgs(X.numpy(), y)
