from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import math
from math import log
import matplotlib
import matplotlib.pyplot as plt

def convert_from_scikit_learn_to_dic_ite(node_index,is_leaves, children_left,children_right,feature,threshold,value,labels,C):

        a = is_leaves[0]
        b = feature[0]
        c = threshold[0]
        if (a):
            d = value[0]  #datapoints of each class in the node
            d2 = np.squeeze(d/np.sum(d))
            d3 = np.zeros(C)
            d3[labels] = d2
            e = labels[np.argmax(d2)]
            return {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True,
            'prediction': e,
            'labels_distribution':d3}

        else:
            left = children_left[0]-node_index[0]
            if(left==-1):
                left_tree = None
            else:
                left_tree = convert_from_scikit_learn_to_dic_ite(node_index[left:],is_leaves[left:], children_left[left:],children_right[left:],feature[left:],threshold[left:],value[left:],labels,C)
            right = children_right[0]-node_index[0]
            if(right==-1):
                right_tree = None
            else:
                right_tree = convert_from_scikit_learn_to_dic_ite(node_index[right:],is_leaves[right:], children_left[right:],children_right[right:],feature[right:],threshold[right:],value[right:],labels,C)
            return {'is_leaf'          : False,
            'prediction'       : None,
            'splitting_feature': b,
            'threshold'        : c,
            'left'             : left_tree,
            'right'            : right_tree,
            'labels_distribution': None}


def convert_from_scikit_learn_to_dic(tree,threshold,C,Q):
    # C is the size of the whole labels
    # labels are the labels that are used in the this tree
    labels = range(0,C,1)
    n_nodes = tree.tree_.node_count
    #n_nodes = subset.shape[0]
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    node_index = np.array(range(0,n_nodes))
    Val = Q   #datapoints in node
# The tree structure can be traversed to compute various properties such
# as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
    # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    return convert_from_scikit_learn_to_dic_ite(node_index,is_leaves, children_left,children_right,feature,threshold,Val,labels,C)

#def TransferToSKLsequence(T):
#    children_left = []
#    children_right = []
#    for i in range(len(T)):



def kl (p,q): # Kullback-libler divegence
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0,p * np.log10((p / q)), 0))

def jsd(p,q): # Symmetric Kullback-libler divergence
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    m = (p+q)/2
    return (kl(p,m)+kl(m,q))/2

def infogain(yleft,len_left,yright,len_right):
    yparent = (yleft+yright)/2
    N = len_left+len_right
    #compute information gain
    I = entropy(yparent) -( (len_left/N)*entropy(yleft) + (len_right/N)*entropy(yright) )
    return I

#entropy for multiple classes
def entropy(y):
    y1 = y[y!=0]
    H = -(y1*np.log10(y1)).sum()
    return H

def partition(Xtarget,ytarget,index_of_data,feature,C,threshold): # divide the data to the left and rightbased on the threshold
    left = index_of_data[Xtarget[index_of_data,feature]<threshold]
    if(len(left)==0):
        left = index_of_data[Xtarget[index_of_data,feature]<=threshold]
    labels_left = ytarget[left]
    qL = np.bincount(labels_left)
    right = index_of_data[Xtarget[index_of_data,feature]>=threshold]
    labels_right = ytarget[right]
    qR = np.bincount(labels_right)
    qR = np.append(qR,np.zeros(np.max([C-qR.shape[0],0])))
    qL = np.append(qL,np.zeros(np.max([C-qL.shape[0],0])))
    qL = qL/qL.sum()
    qR = qR/qR.sum()
    return [qL,left,qR,right]

def dg(Sleft,lenleft,Sright,lenright,QL,QR): # DG function as in the paper
    return 1-(lenleft/(lenleft+lenright))*jsd(Sleft,QL)-(lenright/(lenleft+lenright))*jsd(Sright,QR)

def threshold_selection(X,y,S,f,QL,QR,C,verbos): # finding the best threshold
    fvals = np.sort(X[S,f])
    num_data_points = len(fvals)
    N = 50
    Val  = np.array([])
    #Val_swap  = np.array([])
    Val_infogain = np.array([])
    if num_data_points > N-1:
        I = range(0,num_data_points,np.floor(num_data_points/N).astype(int))
        fvals = fvals[I[1:-1]]
    for i in fvals:
        [Sleft, left, Sright, right] = partition(X,y,S,f,C,i)
        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        #ax1.plot(QL)
        #ax1.set_title('QL')
        #ax2.plot(Sleft, color='r')
        #ax2.set_title('QprimeL')
        #ax3.plot(QR)
        #ax3.set_title('QR')
        #ax4.plot(Sright, color='r')
        #ax4.set_title('QprimeR')
        Val = np.append(Val,dg(Sleft,len(left),Sright,len(right),QL,QR))
        Val_infogain = np.append(Val_infogain,infogain(Sleft,len(left),Sright,len(right)))        #Val_swap = np.append(Val_swap,dg(Sleft,len(left),Sright,len(right),QR,QL)) # this is the divergence measure for each threshold split
    if(verbos):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row')
    #ax1.plot(Val)
    #ax1.set_title('DG')
    #ax2.plot(Val_infogain)
    #ax2.set_title('infogain')
        ax1.plot(fvals,Val,'r')
        ax1.hold(True)
        ax1.plot(fvals,Val_infogain)
        ax1.hold(False)
        ax1.set_title('DG and Infogain')
    #plt.show()
    Val[np.isnan(Val)] = min(Val[~np.isnan(Val)])
    Val_infogain[np.isnan(Val_infogain)] = min(Val_infogain[~np.isnan(Val_infogain)])
    #Val_swap[np.isnan(Val_swap)] = min(Val_swap[~np.isnan(Val_swap)])
    th = fvals[np.argmax(Val)]
    th_infogain = fvals[np.argmax(Val_infogain)]
    if(len(S)<50):
        [ql, left, qr, right] = partition(X,y,S,f,C,th_infogain)
    else:
        [ql, left, qr, right] = partition(X,y,S,f,C,th)
    if(verbos):
        ax2.plot(ql)
        ax2.hold(True)
        ax2.plot(qr)
        ax2.hold(False)
        ax2.set_title('Dist Target Data')

        ax3.plot(QL)
        ax3.hold(True)
        ax3.plot(QR)
        ax3.hold(False)
        ax3.set_title('Dist Source Data')
    return [th, ql, qr, left, right]

def classify(tree, x):
    # if the node is a leaf node.
    if tree['is_leaf']:
        return tree['labels_distribution']
    else:
        # split on feature.
        val_split_feature = x[tree['splitting_feature']]
        if val_split_feature < tree['threshold']:
            return classify(tree['left'], x)
        else:
            return classify(tree['right'],x)

def forest_posterior(RF,x):

    T = len(RF)  #the number of trees

    #infer the number of classes
    P0 = classify(RF[0],x)
    C = len(P0)

    Pt = np.zeros((T,C)) #matrix of posteriors from each tree (T x Nclasses)
    Pt[0,:] = P0
    for t in range(len(RF))[1:]:
        Pt[t,:] = classify(RF[t],x)
    return Pt

#classify input based on majority voting of each tree prediction
def forest_classify_majority(RF,x):
        Pt = forest_posterior(RF,x)
        Yt = np.argmax(Pt,axis=1)
        C,unique_counts = np.unique(Yt,return_counts=True) #the id of classes and number of each
        return C[np.argmax(unique_counts)]

#classify input by averaging posteriors
def forest_classify_ensemble(RF,x):
    Pt = forest_posterior(RF,x)
    Pforest = Pt.mean(axis=0)
    ypred = np.argmax(Pt.mean(axis=0))
    return ypred

def evaluate_classification_error(RF, X, y, method = None):
    # Apply the forest_classify(RF, x) to each row in your data
    if method == None:
        ypred = map(lambda x: forest_classify_ensemble(RF,x), X)
        # Once you've made the predictions, calculate the classification error and return it
        mistakes = sum(ypred != y)
        error = mistakes/len(y)
    return error

def value_for_all(estimator,N):
    from scipy.sparse import csr_matrix
    ch_left = estimator.tree_.children_left
    ch_right = estimator.tree_.children_right
    (cl,) = np.where(ch_left!=-1)
    (cr,) = np.where(ch_right!=-1)
    cap = estimator.tree_.capacity
    dis_node = np.zeros((cap,estimator.tree_.n_classes))
    A = np.zeros([cap,cap])
    D = A
    A = csr_matrix(A)
    A[cl,ch_left[cl]] = 1
    A[cr,ch_right[cr]] = 1
    B = A
    C = B
    while(C.sum()!=0):
        C = A*C
        B = B + C
    I,J = B.nonzero()
    D[I,J] = 1
    (I,) = np.where(ch_left==-1)
    dis_node[I,:] = np.squeeze(estimator.tree_.value[I])
    for i in I:
        dis_node[i,:] = dis_node[i,:]/dis_node[i,:].sum()
    (remain1,) = np.where(ch_left!=-1)
    for i in remain1:
        (I,) = np.where(D[i,:]==1)
        dis_node[i,:] = np.sum(np.squeeze(estimator.tree_.value[I]),axis = 0)
        dis_node[i,:] = dis_node[i,:]/dis_node[i,:].sum()
    Dis_node = np.zeros((cap,N))
    Dis_node[:,estimator.classes_.astype(int)] = dis_node
    return Dis_node

def STRUCT(Xsource,ysource,Xtarget,ytarget,n_trees,C,verbos = False):
    # Assumption: ysource has all the labels of the problem
    #estimator = DecisionTreeClassifier(max_features='sqrt',random_state=0,max_depth=2)
    Estimator = RandomForestClassifier(max_features='sqrt',random_state=0,n_estimators=n_trees)
    Estimator = Estimator.fit(Xsource, ysource)
    RF = []
    for rf in range(Estimator.n_estimators):
        estimator = Estimator.estimators_[rf]
        dis_node = value_for_all(estimator,C)
        P = list(np.zeros(estimator.tree_.capacity))
        P[0] = range(len(ytarget))
        Q = list(np.zeros(estimator.tree_.capacity))
        Q[0] = dis_node[0,:]
        thresh = np.zeros(estimator.tree_.capacity)
        remain = [0]
        subset = []
        while(len(remain)!=0):
            i = remain[0]
            LF = estimator.tree_.children_left
            LR = estimator.tree_.children_right
            index_left = LF[i]
            index_right = LR[i]
            if(index_left!=-1):
                QL = dis_node[index_left,:]
                QR = dis_node[index_right,:]
                f = estimator.tree_.feature[i]
                [th, ql, qr, left, right] = threshold_selection(Xtarget,ytarget,np.array(P[i]),f,QL,QR,C,verbos)
                thresh[i] = th
                P[index_left] = left
                P[index_right] = right
                Q[index_left] = ql
                Q[index_right] = qr
                if(len(left)!=0):
                    remain = np.append(remain,index_left)
                if(len(right)!=0):
                    remain = np.append(remain,index_right)
                if(len(left)>10 and len(right)>10):
                    subset.append(i)
                    print i
            remain = remain[1:]
        lf =  LF[subset]
        lr =  LR[subset]
        subset = np.append(subset,lf)
        subset = np.append(subset,lr)
        print subset
        ST = convert_from_scikit_learn_to_dic(estimator,thresh,C,Q)
        RF.append(ST)
    return RF
