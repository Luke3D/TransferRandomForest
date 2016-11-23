from __future__ import division
import numpy as np
import math
from math import log
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#************SER**************************

#******* TREE FUNCTIONS *********
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


def convert_from_scikit_learn_to_dic(tree,labels,C):
    # C is the size of the whole labels
    # labels are the labels that are used in this tree

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    node_index = np.array(range(0,n_nodes))
    Val = tree.tree_.value   #datapoints in node

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

def evaluate_classification_error_tree(tree, X, y):
    if type(y) == np.uint8:# or X.shape[0] < 2:   #if we have only 1 datapoint in X
        P = classify(tree,X)
        prediction = np.argmax(P)
        error = int(prediction != y)
    else:
        # Apply the classify(tree, x) to each row in your data

        P = map(lambda x: classify(tree,x), X)
        P = np.asarray(P)
        prediction = np.argmax(P,axis=1)
        # Once you've made the predictions, calculate the classification error and return it
        mistakes = sum(prediction != y)
        error = mistakes/len(y)

    return error

def intermediate_node_num_mistakes(labels_in_node):

    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0

    C,unique_counts = np.unique(labels_in_node,return_counts=True) #the id of classes and number of each

    return (len(labels_in_node) - unique_counts[np.argmax(unique_counts)])

def datapath(tree, x, branch = 1):
    # if the node is a leaf node.
    if tree['is_leaf']:
        return branch
    else:
        # split on feature.
        split_feature = tree['splitting_feature']
        split_threshold = tree['threshold']

        if x[split_feature] < split_threshold:
            return datapath(tree['left'], x, 2*branch)
        else:
            return datapath(tree['right'],x, 2*branch+1)

#******* SER FUNCTIONS *********
def expansion_reduction_SKL(tree,XT1,yT1,XT2,yT2,C):

    #finding the leaf where each target datapoint ends up
    leavesData1 = map(lambda x: datapath(tree,x), XT1)
    leavesData2 = map(lambda x: datapath(tree,x), XT2)

    Uleaves1 = np.unique(leavesData1)  #the path to each leaf followed by data1
    Uleaves2 = np.unique(leavesData2)  #the path to each leaf followed by data2
    Uleaves = list(set(Uleaves1) & set(Uleaves2)) #leaves reached by both data1 and data2

    #expanding each leaf on the 1st bootstrap replica of target data
    for i in Uleaves:

        ind_data1 = leavesData1==i #indices of datapoints for each leaf
        ind_data2 = leavesData2==i

        if len(ind_data1) < 2:  #do not expand if there is only 1 datapoint
            continue

        estimator = DecisionTreeClassifier(max_features='sqrt') #use sqrt the number of feat.
        estimator = estimator.fit(XT1[ind_data1,:],yT1[ind_data1])
        #print 'Nclasses ExpTree = %s'%np.unique(yT1[ind_data1]) #%estimator.tree_.n_classes

        Exp_tree = convert_from_scikit_learn_to_dic(estimator,np.unique(yT1[ind_data1]),C)

        #Is this a good expansion?: computes classification error at each leaf for Data T2
        Err_leavesT2 = intermediate_node_num_mistakes(yT2[ind_data2])/len(yT2[ind_data2])

        #error at the current subtree on Data T2
        Err_subtreeT2 = evaluate_classification_error_tree(Exp_tree, XT2[ind_data2,:], yT2[ind_data2])
        #comparing the error of the subtree with that at the leaf node of the original tree
        if Err_subtreeT2 < Err_leavesT2:
            tree = mergetrees(tree,i,Exp_tree)
            #print 'merging successful!'
            #print 'subtree nodes = %s'%count_nodes(Exp_tree)

        #else:
            #print 'no merging: discard subtree'

    return tree

def mergetrees(tree1,leafnr,tree2):
    leafnrbin = bin(leafnr)[3:]  #path is from the 4th element of the binary on: 0 = go left, 1 = go right
    path = ''
    for i in range(len(leafnrbin)):
        if leafnrbin[i] == '0':
            path=path+str("['left']")
        else:
            path=path+str("['right']")
    # print(path)
    exec ('tree1'+path+"['prediction']"+'=None')
    exec ('tree1'+path+"['is_leaf']"+'=False')
    exec ('tree1'+path+"['left']"+"=tree2['left']")
    exec ('tree1'+path+"['right']"+"=tree2['right']")
    exec ('tree1'+path+"['splitting_feature']"+"=tree2['splitting_feature']")
    exec ('tree1'+path+"['threshold']"+"=tree2['threshold']")
    exec ('del(tree1'+path+"['labels_distribution'])")
    return tree1

#**** FOREST FUNCTIONS *******
#Convert Scikit Learn RF to our format
def forest_convert(estimator):
    RF = []  #the new RF list
    ntrees = estimator.n_estimators
    classes = estimator.classes_
    Nclasses = len(classes)
    for t in range(ntrees):
        tree = estimator.estimators_[t] #Scikit learn tree
        Newtree = convert_from_scikit_learn_to_dic(tree,labels=classes,C=Nclasses) #converts to dictionary and saves to list
        RF.append(Newtree)
    return RF

    #outputs the posterior prob of each tree and the corresponding class
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

#Refine the trained forest on target data using the SER algorithm
#Works on forest converted from Scikit Learn to our format
def forest_SER(RF,XT,yT,C,Verbose=False):

    nptrain = len(yT) #how many datapoints each tree is trained (same size of yT)
    ntrees = len(RF)
    RFnew = []

    for t in range(ntrees):
        if Verbose:
            print 'expanding/reducing tree = %s'%t
        #Bootstrap XT1 and XT2
        indbootstrap1 = np.random.choice(XT.shape[0],nptrain)
        indbootstrap2 = np.random.choice(XT.shape[0],nptrain)
        XT1 = XT[indbootstrap1,:]
        XT2 = XT[indbootstrap2,:]
        yT1 = yT[indbootstrap1]
        yT2 = yT[indbootstrap2]

        treeNew = expansion_reduction_SKL(RF[t],XT1,yT1,XT2,yT2,C)
        RFnew.append(treeNew)

    if Verbose:
        print 'Forest refined on target data!'
    return RFnew


#************STRUT functions**************************

def convert_from_scikit_learn_to_dic_ite_strut(node_index,is_leaves, children_left,children_right,feature,threshold,value,labels,C):
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
                left_tree = convert_from_scikit_learn_to_dic_ite_strut(node_index[left:],is_leaves[left:], children_left[left:],children_right[left:],feature[left:],threshold[left:],value[left:],labels,C)
            right = children_right[0]-node_index[0]
            if(right==-1):
                right_tree = None
            else:
                right_tree = convert_from_scikit_learn_to_dic_ite_strut(node_index[right:],is_leaves[right:], children_left[right:],children_right[right:],feature[right:],threshold[right:],value[right:],labels,C)
            return {'is_leaf'          : False,
            'prediction'       : None,
            'splitting_feature': b,
            'threshold'        : c,
            'left'             : left_tree,
            'right'            : right_tree,
            'labels_distribution': None}

def convert_from_scikit_learn_to_dic_strut(feature,threshold,C,Q,children_left,children_right):
    # C is the size of the whole labels
    # labels are the labels that are used in the this tree
    labels = range(0,C,1)
    n_nodes = len(children_left)
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

    return convert_from_scikit_learn_to_dic_ite_strut(node_index,is_leaves, children_left,children_right,feature,threshold,Val,labels,C)

def treesubset(subset,children_left,children_right):
    ch_left = np.zeros(len(children_left))
    ch_right = np.zeros(len(children_right))
    for i in range(len(subset)):
        (l,) = np.where(subset==children_left[i])
        (r,) = np.where(subset==children_right[i])
        if(l.shape[0]==0):
            ch_left[i] = -1
        else:
            ch_left[i] = l

        if(r.shape[0]==0):
            ch_right[i] = -1
        else:
            ch_right[i] = r
    return (ch_left,ch_right)

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
    N = len_left+len_right
    yparent = (len_left/N)*yleft+(len_right/N)*yright
    #compute information gain
    I = entropy(yparent) -( (len_left/N)*entropy(yleft) + (len_right/N)*entropy(yright) )
    return I

#entropy for multiple classes
def entropy(y):
    y1 = y[y!=0]
    H = -(y1*np.log10(y1)).sum()
    return H

#computes split for given feature
def partition(Xtarget,ytarget,index_of_data,feature,num_class_target,index_class_target,threshold): # divide the data to the left and rightbased on the threshold
    left = index_of_data[Xtarget[index_of_data,feature]<threshold]
    if(len(left)==0):
        left = index_of_data[Xtarget[index_of_data,feature]<=threshold]
    labels_left = ytarget[left]
    right = index_of_data[Xtarget[index_of_data,feature]>=threshold]
    labels_right = ytarget[right]

    qL_full = np.bincount(labels_left)
    qR_full = np.bincount(labels_right)
    qL_full = np.append(qL_full,np.zeros(np.max([num_class_target-qL_full.shape[0],0])))
    qR_full = np.append(qR_full,np.zeros(np.max([num_class_target-qR_full.shape[0],0])))

    qL = qL_full[index_class_target]
    qR = qR_full[index_class_target]

    qL = qL/qL.sum()
    qR = qR/qR.sum()

    return [qL,left,qR,right]

def dg(Sleft,lenleft,Sright,lenright,QL,QR): # DG function as in the paper
    return 1-(lenleft/(lenleft+lenright))*jsd(Sleft,QL)-(lenright/(lenleft+lenright))*jsd(Sright,QR)

#optimize threshold of given node based on target data
#INPUTS: Xtarget, ytarget, S:indices of datapoints that reach node, f:feature of node, QL,QR:Distributions from source reaching children
#        C:# of classes, verbos: print additional info
#OUTPUTS: [th, ql, qr, left, right]: th: new threshold; ql,qr: distribution from target with new threshold; left,right: target data indices
#         going to left and right nodes
#Depending fcns: partition(), df(), infogain(), jsd(), kl(), entropy()

def threshold_selection(X,y,S,f,QL,QR,num_class_target,index_class_target,verbos): # finding the best threshold
    fvals = np.sort(X[S,f])
    num_data_points = len(fvals)
    N = 50 # # of bins to search threshold
    Val_DG  = np.array([]) #contains values of DG for each bin
    Val_infogain = np.array([])
    if num_data_points > N-1:
        I = range(0,num_data_points,np.floor(num_data_points/N).astype(int))
        fvals = fvals[I[1:-1]]
    for i in fvals:
        [qL, left, qR, right] = partition(X,y,S,f,num_class_target,index_class_target,i) #compute histogram (qL,qR) for left and right children with current threshold
        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
        #ax1.plot(QL)
        #ax1.set_title('QL')
        #ax2.plot(Sleft, color='r')
        #ax2.set_title('QprimeL')
        #ax3.plot(QR)
        #ax3.set_title('QR')
        #ax4.plot(Sright, color='r')
        #ax4.set_title('QprimeR')
        Val_DG = np.append(Val_DG,dg(qL,len(left),qR,len(right),QL,QR))
        Val_infogain = np.append(Val_infogain,infogain(qL,len(left),qR,len(right)))        #Val_swap = np.append(Val_swap,dg(Sleft,len(left),Sright,len(right),QR,QL)) # this is the divergence measure for each threshold split
    if(verbos):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex='col', sharey='row')
    #ax1.plot(Val)
    #ax1.set_title('DG')
    #ax2.plot(Val_infogain)
    #ax2.set_title('infogain')
        ax1.plot(fvals,Val_DG,'r')
        ax1.hold(True)
        ax1.plot(fvals,Val_infogain)
        ax1.hold(False)
        ax1.set_title('DG and Infogain')
    #plt.show()
    #find nan and assign the minimum value found in the array to remove the nan
    Val_DG[np.isnan(Val_DG)] = min(Val_DG[~np.isnan(Val_DG)])
    Val_infogain[np.isnan(Val_infogain)] = min(Val_infogain[~np.isnan(Val_infogain)])
    #Val_swap[np.isnan(Val_swap)] = min(Val_swap[~np.isnan(Val_swap)])
    th_DG = fvals[np.argmax(Val_DG)]
    th_infogain = fvals[np.argmax(Val_infogain)]

    #plt.plot(Val_infogain)
    #plt.show()

    #find new threshold based on how many datapoints we have in current node
    #Set DG vs IG
    if(len(S)>0):
        [ql, left, qr, right] = partition(X,y,S,f,num_class_target,index_class_target,th_infogain)
        th = th_infogain
    else:
        [ql, left, qr, right] = partition(X,y,S,f,num_class_target,index_class_target,th_DG)
        th = th_DG

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


#returning the histogram of the classes for each node in estimator. N is # of classes
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

def STRUT(Xsource,ysource,Xtarget,ytarget,n_trees,verbos = False):
    # Assumption: ysource has all the labels of the problem
    #Train RF on Source
    Estimator = RandomForestClassifier(n_estimators=n_trees,criterion='entropy',random_state=0)
    Estimator = Estimator.fit(Xsource, ysource)
    #Infer classes of source and target
    C = len(np.unique(ysource)) # Number of classes in source data
    index_class_target = np.sort(np.unique(ytarget)) #the indices of classes in the target data
    max_index_target = max(index_class_target)+1

    # ypred = Estimator.predict(Xtarget)
    # print sum(ypred!=ytarget)/len(ytarget)

    RF = []

    # Looping through all the trees in the forest RF
    for rf in range(Estimator.n_estimators):
        estimator = Estimator.estimators_[rf]
        #print '#leaves before STRUT = %s'%(estimator.tree_.node_count)

        dis_node = value_for_all(estimator,C)  #Histogram of classes for source (row: node index, col: class)

        #if the target data contains a different # of classes than source
        if len(index_class_target) < C:
            dis_node = dis_node[:,index_class_target]/np.sum(dis_node[:,index_class_target],axis=1)[:,None]
        dis_node[np.isnan(dis_node)] = 1/len(index_class_target)
        #if(np.any(np.isnan(dis_node))):
         #   print 'nan found'

        LF = estimator.tree_.children_left  #indices of left nodes
        LR = estimator.tree_.children_right #indices of right nodes
        Features = estimator.tree_.feature
        num_nodes = estimator.tree_.capacity
        P = list(np.zeros(num_nodes)) #maintain a list of the target data indices at each node (0 is root) - capacity is node_count
        P[0] = range(len(ytarget))
        Q = np.zeros((num_nodes,len(index_class_target))) #Histogram of target data indices at each node
        Q[0,:] = dis_node[0,:]
        thresh = np.zeros(num_nodes) #init vector with new threshold at each node
        remain = [0]    #indexing the remaining nodes as we move from top-down
        subset = []     #maintains a list of the nodes that are reached from target data
        #looping through the nodes
        while(len(remain)!=0):
            i = remain[0]
            subset.append(i)  #updating the list of nodes reached by target data with current node index
            index_left = LF[i]
            index_right = LR[i]
            #check if node is leaf
            if(index_left!=-1):
                QL = dis_node[index_left,:]  #distribution of labels in children nodes
                QR = dis_node[index_right,:]
                f = Features[i]  #feature of parent node
                [th, ql, qr, left, right] = threshold_selection(Xtarget,ytarget,np.array(P[i]),f,QL,QR,max_index_target,index_class_target,verbos)
                thresh[i] = th
                P[index_left] = left
                P[index_right] = right
                Q[index_left,:] = ql
                Q[index_right,:] = qr

                #if no target datapoints reach either the left or right children we make the parent node a leaf
                if(len(left)>0 and len(right)>0):
                    remain = np.append(remain,index_left)
                    remain = np.append(remain,index_right)
            remain = remain[1:]

        subset = np.sort(subset)
        lf =  LF[subset]
        lr =  LR[subset]
        (ch_lf,ch_lr) = treesubset(subset,lf,lr)
        Qnew = np.zeros((Q.shape[0],C))
        Qnew[:,index_class_target] = Q  #Histogram of target data on the full set of class labels (includes the missing classes)
        ST = convert_from_scikit_learn_to_dic_strut(Features[subset],thresh[subset],C,Qnew[subset,:],ch_lf.astype(int),ch_lr.astype(int))
        #print 'tree trained'
        RF.append(ST)
        #print '#leaves after STRUT = %s'%(sum(lf==-1))

    return RF
