import sys
import numpy as np
import math

##functions

##read the data
def readData(path):
    p = path
    with open(path, "r") as f:
        data = []
        labels = []
        for idx,line in enumerate(f):
            if idx == 0:
                header = line.strip().split(',')
            else:
                line = line.strip().split(',')
                labels.append(line[-1])                
                data.append(line)

            train_data = np.array(data)

    return header, train_data, labels

def entropy(labels):
    totalNumofLabels = len(labels)
#     if total <= 1:
#         return 0
    classes, countOfClasses = np.unique(labels, return_counts=True)
    if len(classes) <= 1: #if arity of the labels is 1 then the entropy is 1
        return 0
    ##computation
    probs = countOfClasses/totalNumofLabels
    ent = 0
    for p in probs:
        ent += p*math.log2(p) #calculation used to determine the entropy of the labels     
    return -1 * ent

def mutualInfo(data, position):
    ent = entropy(data[:,-1])
    lengthOfValues = len(data[:, position])
    classes, countOfClasses = np.unique(data[:, position], return_counts=True)
    if len(classes) <= 1:
        return 0
    ##calculate probabilities used in conditional entropy
    probs = countOfClasses/lengthOfValues
    ##calculate specific conditional entropies
    specCondEnts = []
    for c in classes:
        specCondEnt = 0
        boolean = data[:, position] == c
        subset = data[boolean]
        lengthOfSubset = len(subset)
        subsetClasses, subsetCountofClasses = np.unique(subset[:,-1], return_counts=True)
        subsetClasses, subsetCountofClasses
        probsOfSubset = subsetCountofClasses/lengthOfSubset
        for p in probsOfSubset:
            if p != 0:
                specCondEnt += p*math.log2(p)
            else:
                specCondEnt += 0
        specCondEnts.append(-1*specCondEnt)
    #calculate conditional entropy
    condEnt = sum(p*specCondEnts[idx] for idx, p in enumerate(probs))
    #calculate info gain
    infoGain = ent - condEnt
    return infoGain

#function used to split the data at each node
def splitData(data, idx):
    classes = np.unique(data[:,idx])
    subsets = []
    for c in classes:
        boolean = data[:, idx] == c
        subset = data[boolean]
        subsets.append(subset)

    return subsets   

#function returns attribute that obtains the highest mutual information at each node
def bestMutualInfo(data, attributes): #optimize this to work better
    if len(attributes) == 1: #unnecessary
        return attributes[0]
    elif len(attributes) == 0:
        return None
    else:
        atts = attributes.copy()[:len(attributes)-1]
        bestInfoGain = -np.inf
        bestatt = None
        for idx, att in enumerate(atts):
#             idx = header.index(att)
            if mutualInfo(data, idx) >= bestInfoGain: 
                bestInfoGain = mutualInfo(data, idx)
#                 bestatt = header[idx]
                bestatt = atts[idx]

        return bestInfoGain, bestatt

#need to pass last column which is the target variable # calculates the majority class for the labels
def maj_classifer(data): 
    labels = data.copy()
    classes, countOfClasses = np.unique(labels, return_counts=True)
    if len(classes) == 1:
        return classes[0]
    counter = 0
    maxValue = -np.inf
    best_label = None
    while counter < len(classes):
        for idx, count in enumerate(countOfClasses):
            if count >= maxValue:
                maxValue = count
                best_label = classes[idx]
            counter += 1
    return best_label

#Node class used within building the tree
class Node:
    def __init__(self, data, maxdepth):
        self.data = data
        self.maxdepth = maxdepth
        self.key = None #splitting attribute
        self.infogain = None
        self.label = None #classifier
        self.left = None
        self.right = None
        self.parent = None
        self.value = None #attribute value (ex. 'y' or 'n')

#Recursive Algorithm to build the tree
def buildTree(traindata,feats, maxdepth): 
    # maxdepth is 0, then return the majority output class
    if maxdepth == 0:
        return maj_classifer(traindata[:,-1]) 
    #maxdepth is limited by number of features
    if maxdepth > len(feats) + 1: 
        maxdepth = len(feats) + 1
    #create the root node with all the training data and an initial max depth
    root = Node(traindata, maxdepth) 
    #calculate the information gain at that node and the attribute that best splits the data at that node
    infoGainVal, bestAtt = bestMutualInfo(root.data, feats) 
    #base case
    if infoGainVal <= 0: 
        root.key = 'leaf'
        root.label = maj_classifer(root.data[:, -1])
        return root
    root.key = bestAtt
    root.label = maj_classifer(root.data[:,-1])
    root.infogain = infoGainVal
    #split the root node data 
    rightData, leftData = splitData(root.data, feats.index(bestAtt))

    #recurse to the left subtree
    if maxdepth != 1 and root.left == None :
        root.left = buildTree(leftData, feats, maxdepth - 1) 
        root.left.value = leftData[0,feats.index(bestAtt)]
        root.left.parent = bestMutualInfo(root.data, feats)[1]

    #recurse to the right subtree
    if maxdepth != 1 and root.right == None :
        root.right = buildTree(rightData, feats, maxdepth - 1) 
        root.right.value = rightData[0,feats.index(bestAtt)]
        root.right.parent = bestMutualInfo(root.data, feats)[1]

    return root

#Use recursion to print the tree
def printPreorder(root, classOne, classTwo, counter = 0):
    if root:
        classes = [classOne, classTwo]
        _, countOfClasses = np.unique(root.data[:,-1], return_counts=True)
        if counter == 0:
            print('[{} {} /{} {}]\n'.format(countOfClasses[0], classes[0], countOfClasses[1], classes[1]))
        else:
            if len(countOfClasses) == 2:
                print('|'* counter + '{} = {}: [{} {} /{} {}] \n'.format(root.parent, root.value, countOfClasses[0], classes[0], countOfClasses[1], classes[1]))
            elif len(countOfClasses) == 1 and _ == classes[0]:
                print('|' * counter +'{} = {}: [{} {} /{} {}] \n'.format(root.parent, root.value, countOfClasses[0], classes[0], 0, classes[1]))
            else:
                print('|' * counter + '{} = {}: [{} {} /{} {}] \n'.format(root.parent, root.value, 0, classes[0], countOfClasses[0], classes[1]))

        # Then recur on left child
        printPreorder(root.left, classes[0], classes[1], counter+1)
        #Finally recur on right child
        printPreorder(root.right, classes[0], classes[1], counter+1)

#Recursive function that traverses the tree and return the prediction of the query
def prediction(tree, feats, row, maxdepth, currentdepth=1):
    #base case
    if tree.key == 'leaf':
        return tree.label
    #base case
    if maxdepth == currentdepth:
        return tree.label
    #recurse
    if any(tree.key == feat for feat in feats):
        idx = feats.index(tree.key)
        if row[idx] == tree.left.value:
            left = prediction(tree.left,feats,row, maxdepth, currentdepth + 1)
            return left
        if row[idx] == tree.right.value:
            right = prediction(tree.right,feats,row, maxdepth, currentdepth + 1)
            return right

def writeOutput(data, tree, feats, output, maxdepth):
    with open(output, 'w') as f:
        for row in data:
            f.write(prediction(tree, feats, row, maxdepth) + '\n')

def metrics(trainCsvFile, trainOutputFile, testCsvFile, testOutputFile, metricfile):
    with open(trainCsvFile, 'r') as csv:
        trainInputData = []
        for line in csv.readlines()[1:]:
            line = line.strip().split(',')
            trainInputData.append(line)

    with open(trainOutputFile, 'r') as out:
        trainOutputData = []
        for line in out.readlines():
            line = line.strip()
            trainOutputData.append(line)

    with open(testCsvFile, 'r') as csv:
        testInputData = []
        for line in csv.readlines()[1:]:
            line = line.strip().split(',')
            testInputData.append(line)

    with open(testOutputFile, 'r') as out:
        testOutputData = []
        for line in out.readlines():
            line = line.strip()
            testOutputData.append(line)


    total = len(trainOutputData)
    counter = 0
    for i, o in zip(trainInputData, trainOutputData):
        i = i[-1]
        if i != o:
            counter += 1

    trainError = counter/total

    total = len(testOutputData)
    counter = 0
    for i, o in zip(testInputData, testOutputData):
        i = i[-1]
        if i != o:
            counter += 1

    testError = counter/total

    with open(metricfile, 'w') as met:
        met.write('error(train): {}\n'.format(trainError))
        met.write('error(test): {}\n'.format(testError))

def main():
    #command line arguments
    traincsv, testcsv, maxDepth, trainOut, testOut, metricsOut  = sys.argv[1:]
    if int(maxDepth) > 0:
         maxDepth = int(maxDepth) + 1
    else:
        maxDepth = int(maxDepth)
    #reading the train and test data in
    trainAttributes, trainInput, trainLabels = readData(traincsv)
    testAttributes, testInput, testLabels = readData(testcsv)
    #Build the decision tree on train data
    root = buildTree(trainInput, trainAttributes, maxDepth) #d, h, depth
    #create train.labels file
    writeOutput(trainInput, root, trainAttributes, trainOut, maxDepth)
    #create test.labels file
    writeOutput(testInput, root, testAttributes, testOut, maxDepth)
    #create training metric file
    metrics(traincsv, trainOut, testcsv, testOut, metricsOut)
    
    #create testing metric file
    # metrics(testcsv, testOut, metricsOut, 'test')

    #printout of trained tree
    classes = np.unique(root.data[:,-1])
    printPreorder(root, classes[0], classes[1])

if __name__ == "__main__":
    main()

