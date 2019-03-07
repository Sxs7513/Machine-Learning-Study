class Graph(object):
    def __init__(self):
        self.operations, self.constants, self.placeholders = [], [], []
        self.variables, self.trainable_variables = [], []

    def __enter__(self):
        global DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH
        DEFAULT_GRAPH = self
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        global DEFAULT_GRAPH
        DEFAULT_GRAPH = self.old_graph

    def as_default(self):
        return self
