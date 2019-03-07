from functools import reduce

from .operations import Operation, Variable, Placeholder

class Session(object):

    def __init__(self):
        # Graph the session computes for.
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    def close(self):
        ''' Free all output values in nodes.
        '''
        all_nodes = (self.graph.constants + self.graph.variables +
                     self.graph.placeholders + self.graph.operations +
                     self.graph.trainable_variables)
        for node in all_nodes:
            node.output_value = None

    # run 的时候会深度优先搜索, 从底至上将 output_value 全部找到
    # 然后会输出 target 的输出值
    def run(self, operation, feed_dict=None):
        # Get all prerequisite nodes using postorder traversal.
        postorder_nodes = _get_prerequisite(operation)

        for node in postorder_nodes:
            if type(node) is Placeholder:
                node.output_value = feed_dict[node]
            else:  # Operation and variable
                node.compute_output()

        return operation.output_value

def _get_prerequisite(operation):
    postorder_nodes = []

    # Collection nodes recursively.
    def postorder_traverse(operation):
        if isinstance(operation, Operation):
            for input_node in operation.input_nodes:
                postorder_traverse(input_node)
        postorder_nodes.append(operation)

    postorder_traverse(operation)

    return postorder_nodes