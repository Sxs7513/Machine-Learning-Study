# http://pytlab.github.io/2018/01/25/%E5%AE%9E%E7%8E%B0%E5%B1%9E%E4%BA%8E%E8%87%AA%E5%B7%B1%E7%9A%84TensorFlow-%E4%BA%8C-%E6%A2%AF%E5%BA%A6%E8%AE%A1%E7%AE%97%E4%B8%8E%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD/

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

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


# ------------------------------------------------------------------------------
# Addition operation
# ------------------------------------------------------------------------------
class Add(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)
    
    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value
    
    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)
        
        grad_wrt_x = grad
        # 如果梯度的维度大于输入x那么代表y的维度大于x导致输出的时候产生了广播
        # 那么从0维不断累加直到维度相同即可，因为广播的时候也总是将 x 从 0 维
        # 起增加的
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        # 广播也可能存在于 x 的内部，当 x shape 里存在 1 的时候，也可能被广播
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
        
        grad_wrt_y = grad
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]
        

def add(x, y, name=None):
    return Add(x, y, name)


# ------------------------------------------------------------------------------
# Multiplication operation
# ------------------------------------------------------------------------------
class Multiply(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        grad_wrt_x = grad*y
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad*x
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]

def multiply(x, y, name=None):
    return Multiply(x, y, name)


# ------------------------------------------------------------------------------
# Matrix multiplication operation
# ------------------------------------------------------------------------------
class MatMul(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x, y = [node.output_value for node in self.input_nodes]

        if not grad:
            grad = np.ones_like(self.output_value)

        dfdx = np.dot(grad, np.transpose())
        dfdy = np.dot(np.transpose(x), grad)

        return [dfdx, dfdy]
         

def matmul(x, y, name=None):
    return MatMul(x, y, name)


# ------------------------------------------------------------------------------
# Negative operation
# ------------------------------------------------------------------------------
class Negative(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = -x.output_value
        return self.output_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output_value)

        return -grad


# ------------------------------------------------------------------------------
# Reduce sum operation
# ------------------------------------------------------------------------------
class ReduceSum(Operation):
    def __init__(self, x, axis=None):
        super(self.__class__, self).__init__(x)
        self.axis = axis
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sum(x.output_value, axis=self.axis)
        return self.output_value

    def compute_gradient(self, grad=None):
        input_value = self.input_nodes[0].output_value

        # 大部分时候都会进入这里,将对该层输出的梯度形状改为与输入一致
        # 不需要其他任何操作. 因为该操作符只是单纯的相加, 对输入的梯度都是 1
        if grad is None:
            grad = np.ones_like(self.output_value)

        output_shape = np.array(np.shape(input_value))
        output_shape[self.axis] = 1.0
        tile_scaling = np.shape(input_value) // output_shape
        # 将传来的梯度 reshape 到和输入一样的 shape, 这里的代码其实是不严谨的
        # 必须保证 grad 的维度满足某些要求才行, 不过幸好该操作符一般就是用于 loss 的
        # 所以 grad 基本都是 None
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)


def reduce_sum(x, axis=None):
    return ReduceSum(x, axis=axis)


# ------------------------------------------------------------------------------
# Square operation
# ------------------------------------------------------------------------------
class Square(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.square(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)
        
        return grad * np.multiply(2.0, input_value)

def square(x, name=None):
    return Square(x, name=name)


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

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

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

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

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

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)

def placeholder(name=None):
    return Placeholder(name=name)


# ------------------------------------------------------------------------------
# Function for gradients computation.
# ------------------------------------------------------------------------------
# 广度优先搜索
def compute_gradients(target_op):
    # 存储所有在 target_op 这条链上面的所有 node 的梯度
    grad_table = {}
    # 顶级的节点木有上级 grad
    grad_table[target_op] = np.ones_like(target_op.output_value)

    queue = Queue()
    queue.put(target_op)

    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()

        if node != target_op:
            grads_wrt_node_output = []
            # 遍历使用了当前结点的节点，取出对该节点的所有梯度
            for output_node in node.output_nodes:
                grad_wrt_output_node_output = grad_table[output_node]
                grad_wrt_node_output = output_node.compute_gradient(grad_wrt_output_node_output)
                # 如果存在多个输入，那么会有多个输出的梯度，只取需要的
                if len(output_node.input_nodes) > 1:
                    input_node_index = output_node.input_nodes.index(node)
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])
                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)

            # 将所有对该节点的梯度加起来，并缓存
            tot_grad_wrt_node_output = sum(grads_wrt_node_output)
            grad_table[node] = tot_grad_wrt_node_output

        # 输入节点依次向下计算梯度
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                # 已经计算过到该 node 的梯度的话就不用再计算了
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table