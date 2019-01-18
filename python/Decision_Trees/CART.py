import pandas as pd
import numpy as np
from pprint import pprint
import csv

df = pd.read_csv(
    './CART.data',
    sep="	",
    names=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Name']
)


class Tree:
    def __init__(self, value=None, trueBranch=None, falseBranch=None, results=None, summary=None, data=None, feature=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.summary = summary
        self.data = data
        self.feature = feature


def splitData(data, value, feature):
    if (isinstance(value, int) or isinstance(value, float)):
        case = data[feature] >= value
    else:
        case = data[feature] == value

    targetData = data.where(case).dropna()
    otherData = data.where(case == False).dropna()
    return (targetData, otherData)


def gini(classes):
    features, counts = np.unique(classes, return_counts=True)
    total = np.sum(counts)
    imp = 0

    for count in counts:
        imp += np.square(count / total)

    return 1 - imp


def buildDecisionTree(data, evalFunc=gini):
    # 计算当前的熵
    curGain = evalFunc(data['Name'])

    # 取出所有特征名
    features = data.columns[:-1]

    best_gain = 0.0
    best_value = None
    best_split = None
    best_choose = None
    # targetData, otherData = splitData(data, 4.8, 'SepalLength')
    # pprint(targetData.shape)
    # pprint(data.shape)

    for feature in features:
        unique_value = np.unique(data[feature])
        for value in unique_value:
            targetData, otherData = splitData(data, value, feature)
            p = targetData.shape[0] / data.shape[0]

            gain = curGain - p * \
                evalFunc(targetData['Name']) - (1 - p) * \
                evalFunc(otherData['Name'])

            if gain > best_gain:
                best_gain = gain
                best_choose = (feature, value)
                best_split = (targetData, otherData)

    dcY = {'impurity': '%.3f' % curGain, 'samples': '%d' % data.shape[0]}

    if best_gain > 0:
        trueBranch = buildDecisionTree(best_split[0], evalFunc)
        falseBranch = buildDecisionTree(best_split[1], evalFunc)
        tree = Tree(
            feature=best_choose[0],
            value=best_choose[1],
            trueBranch=trueBranch,
            falseBranch=falseBranch,
            summary=dcY
        )
    else:
        tree = Tree(
            results=np.unique(data['Name'], return_counts=True),
            data=data,
            summary=dcY
        )

    return tree


def plot(decisionTree):
    """Plots the obtained decision tree. """

    def toString(decisionTree, indent=''):
        if decisionTree.results != None:  # leaf node
            elements, counts = decisionTree.results
            obj = {}
            for i, element in enumerate(elements):
                obj[element] = counts[i]

            return str(obj)
        else:
            case = 'Column %s' % decisionTree.feature

            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = '%s >= %s?' % (case, decisionTree.value)
            else:
                decision = '%s == %s?' % (case, decisionTree.value)
            trueBranch = indent + 'yes -> ' + \
                toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no  -> ' + \
                toString(decisionTree.falseBranch, indent + '\t\t')
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))


def prune(tree, minGain, evalFunc=gini):
    if tree.trueBranch.results == None:
        prune(tree.trueBranch, minGain, evalFunc)
    if tree.falseBranch.results == None:
        prune(tree.falseBranch, minGain, evalFunc)

    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        len1 = tree.trueBranch.data.shape[0]
        len2 = tree.falseBranch.data.shape[0]
        len3 = pd.concat([tree.trueBranch.data, tree.falseBranch.data])
        p = float(len1) / (len1 + len2)

        gain = evalFunc(
            pd.concat([tree.trueBranch.data, tree.falseBranch.data])['Name'])
        - p * evalFunc(tree.trueBranch.data['Name'])
        - (1 - p) * evalFunc(tree.falseBranch.data['Name'])

        # 发现 gain 小于阈值，代表该分叉没有达到期望的效果，则合并分支
        if (gain < minGain):
            tree.data = pd.concat(
                [tree.trueBranch.data, tree.falseBranch.data])
            tree.results = np.unique(tree.data['Name'], return_counts=True)
            tree.trueBranch = None
            tree.falseBranch = None


def classify(data, tree):
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.feature][0]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if v == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        return classify(data, branch)


tree = buildDecisionTree(df)
plot(tree)
prune(tree, 0.4)
plot(tree)
print(classify(
    pd.DataFrame(
        [[5.0, 2.3, 3.3, 1.0]],
        columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    ),
    tree
))
