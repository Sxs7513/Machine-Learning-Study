# https://segmentfault.com/a/1190000013359859
# http://www.fdlly.com/p/207551054.html
# 随机树森林
import pandas as pd
import numpy as np
import random
# np.set_printoptions(threshold=np.inf)


def cross_validation_split(dataset, n_folds):
    # 因为要分出来测试集, 所以加一
    n_folds = n_folds + 1
    randomIndex = random.sample(range(dataset.shape[0]), len(dataset))
    dataset = dataset[randomIndex]
    
    partLen = int(dataset.shape[0] / n_folds)
    dataset_split = []

    for i in range(n_folds):
        dataset_split.append(dataset[i * partLen : (i + 1) * partLen, :])

    dataset_split[-1] = np.concatenate([dataset[:(dataset.shape[0] - partLen * n_folds), :], dataset_split[-1]], axis=0)

    return np.array(dataset_split)


def accuracy_metric(actual, predicted):
    return np.sum((actual == predicted).astype("int32")) / actual.shape[0] * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []

    for fold in range(n_folds):
        train_set = folds.copy()
        test_set = folds[fold].copy()
        train_set = np.delete(train_set, fold, 0)

        actual = test_set[:, -1].copy()
        # 测试集标签置为 None
        test_set[:, -1] = None
        
        predicted = algorithm(train_set, test_set, *args)
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


def try_split(featureIndex, rowIndex, dataset):
    value = dataset[rowIndex, featureIndex]
    
    indices1 = np.where(dataset[:, featureIndex] >= value)[0]
    indices2 = np.where(dataset[:, featureIndex] < value)[0]

    left = dataset[indices1]
    right = dataset[indices2]

    return left, right


def gini_single(classes):
    features, counts = np.unique(classes, return_counts=True)
    total = np.sum(counts)
    imp = 0

    for count in counts:
        imp += np.square(count / total)

    return 1 - imp


# classes 是 split 之前的分类结果
def cal_gini(groups, classes):
    # 当前的基尼系数
    cur_gini = gini_single(classes)
    
    gini = 0

    left = groups[0][:, -1]
    right = groups[1][:, -1]
    
    p1 = left.shape[0] / classes.shape[0]
    p2 = right.shape[0] / classes.shape[0]
    
    return gini_single(left) * p1 + gini_single(right) * p2
        
    
def get_split(dataset, n_features):
    # split 之前的分类结果
    class_values = dataset[:, -1]
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    
    for featureIndex in range(dataset.shape[1] - 1):
        for rowIndex in range(dataset.shape[0]):
            groups = try_split(featureIndex, rowIndex, dataset) 
            gini = cal_gini(groups, class_values)
            
            if gini < b_score:
                b_index, b_value, b_score, b_groups = featureIndex, dataset[rowIndex, featureIndex], gini, groups
               
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    cates = group[:, -1].tolist()
    
    # 返回最多的类别
    return np.argmax(np.bincount(cates))


def split(node, max_depth, min_size, n_features, depth):
    left, right = node["groups"]
    del(node['groups'])

    if left.shape[0] == 0 or right.shape[0] == 0:
        final = left if left.shape[0] > 0 else right
        node['left'] = node['right'] = to_terminal(final)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if left.shape[0] <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth + 1)
    
    if right.shape[0] <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth + 1)


def build_tree(train, max_depth, min_size, n_features):
    # 构造根节点
    root = get_split(train, n_features)
    # 然后递归  
    split(root, max_depth, min_size, n_features, 1)

    return root


def predict(node, row):
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]

    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return np.argmax(np.bincount(predictions))


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = train[i]
        tree = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)

    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


if __name__ == "__main__":
    filename = 'sonar.all-data.csv'
    df = pd.read_csv(
        filename,
        sep=",",
        header=None,
    ).values

    data = df[:, :-1]
    cates = df[:, -1]
    data = data.astype("float32")
    catesUnique = np.unique(cates)
    for i, cate in enumerate(catesUnique):
        indices = np.where(cates == cate)
        cates[indices] = i

    dataset = np.concatenate([data, cates[:, np.newaxis]], axis=1)
    
    n_folds = 10
    max_depth = 15
    min_size = 1
    sample_size = 1.0
    # 
    n_features = np.sqrt([data.shape[1]]).astype("int32")
    scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_folds, n_features)
    # print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))