from __future__ import division
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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
