from queue import Queue

import numpy as np

class Operation(object):
    def __init__(self, *input_nodes, name=None):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.output_value = None
        self.name = name
        self.graph = DEFAULT_GRAPH

        # 将当前节点的引用添加到他输入节点的output_nodes这样可以在输入节点中找到当前节点
        for node in input_nodes:
            node.output_nodes.append(self)

        # 将当前节点的引用添加到图中，方便后面对图中的资源进行回收等操作
        self.graph.operations.append(self)

    def compute_output(self):
        raise NotImplementedError

    def compute_gradient(self, grad=None):
        raise NotImplementedError


# ------------------------------------------------------------------------------
# Addition operation
# ------------------------------------------------------------------------------
class Add(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)
    
    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)


# ------------------------------------------------------------------------------
# Matrix multiplication operation
# ------------------------------------------------------------------------------
class 


# ------------------------------------------------------------------------------
# Constant node
# ------------------------------------------------------------------------------
class Constant(object):
    def __init__(self, value, name=None):
        self.value = value
        self.output_value = None
        self.output_nodes = []
        self.name = name
        DEFAULT_GRAPH.constants.append(self)

    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.value
        return self.output_value

def constant(value, name=None):
    return Constant(value, name=name)

# ------------------------------------------------------------------------------
# Variable node
# ------------------------------------------------------------------------------
class Variable(object):
    def __init__(self, initial_value=None, name=None, trainable=True):
        self.initial_value = initial_value
        self.output_value = None
        self.output_nodes = []
        self.name = name
        self.graph = DEFAULT_GRAPH
        self.graph.variables.append(self)
        if trainable:
            self.graph.trainable_variables.append(self)

    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.initial_value
        return self.initial_value


# ------------------------------------------------------------------------------
# Placeholder node
# ------------------------------------------------------------------------------
class Placeholder(object):
    def __init__(self, name=None):
        self.output_value = None
        self.output_nodes = []
        self.name = name
        self.graph = DEFAULT_GRAPH
        self.graph.placeholders.append(self)

def placeholder(name=None):
    return Placeholder(name=name)