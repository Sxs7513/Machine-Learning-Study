from .operations import Operation, compute_gradients

class GradientDescentOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            # 在计算输出阶段来进行梯度下降
            def compute_output(self):
                grad_table = compute_gradients(loss)

                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]

                    var.output_value -= learning_rate * grad

        return MinimizationOperation()