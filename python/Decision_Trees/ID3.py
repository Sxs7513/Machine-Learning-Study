import pandas as pd
import numpy as np
from pprint import pprint

label_names = ['animal_name', 'hair', 'feathers', 'eggs', 'milk',
               'airbone', 'aquatic', 'predator', 'toothed', 'backbone',
               'breathes', 'venomous', 'fins', 'legs', 'tail', 'domestic', 'catsize', 'class', ]

dataset = pd.read_csv(
    './ID3.data',
    names=label_names
)
dataset = dataset.drop('animal_name', axis=1)

# 前八十行定为训练集，剩下的定为预测集


def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data, testing_data


training_data, testing_data = train_test_split(dataset)


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([
        (-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts))
        for i in range(len(elements))
    ])

    return entropy


def InfoGain(data, split_attribute_name, target_name="class"):
    total_entropy = entropy(data[target_name])

    # 获得当前数据集在该属性下的所有值(unique过的), 并获得它们各自的数量
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    Weighted_Entropy = np.sum([
        (counts[i] / np.sum(counts)) *
        entropy(
            # 下面代码用于找到当前数据集下,某属性下其中一个值所对应的所有 class
            # where 接受一个dataFrame, 某一行判断为 true 则返回该行原始值, 否则返回全为 NaN 的行
            data.where(data[split_attribute_name] ==
                       vals[i]).dropna()[target_name]
        )
        for i in range(len(vals))
    ])

    Information_Gain = total_entropy - Weighted_Entropy

    return Information_Gain


def ID3(data, originaldata, features, target_attribute_name="class", parent_node_class=None):
    # 当前数据集下只有一类了,则将该 class 返回(class 对应的数字)
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    # 当前数据集为空了,那么存在该情况吗?
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(
            np.unique(originaldata[target_attribute_name],
                      return_counts=True)[1]
        )]
    # 如果分到不能再分了,那么结束该分支
    elif len(features) == 0:
        return parent_node_class

    # 计算当前的数据集下类别最多的是哪种(比如 class 1)
    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(
                np.unique(data[target_attribute_name], return_counts=True)[1]
            )
        ]

        # 计算当前数据集下能最佳划分的属性
        item_values = [
            InfoGain(data, feature, target_attribute_name)
            for feature in features
        ]

        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            value = value
            # 找到当前最优属性下的每个类型的数据集
            sub_data = data.where(data[best_feature] == value).dropna()
            # 递归子数据集
            subtree = ID3(sub_data, dataset, features,
                          target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

    return tree


def predict(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result


def test(data, tree):
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class"])/len(data))*100,'%')


tree = ID3(training_data,training_data,training_data.columns[:-1])
pprint(tree)
test(testing_data,tree)