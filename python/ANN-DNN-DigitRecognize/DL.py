import numpy as np

reg = 0.1

def relu(z):
    s = np.maximum(0, z)
    cache = z

    return s, cache

def softmax(z):
    cache = z
    # 
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache

def initialize_parameters_deep(dims):
    np.random.seed(3)
    params = {}

    for i in range(1, len(dims)):
        # 方便在前向传播的时候直接可以 w dot a
        params['W' + str(i)] = np.random.randn(dims[i], dims[i-1]) * 0.01
        params['b' + str(i)] = np.zeros((dims[i], 1))
    return params 

def compute_cost(A, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(A))
    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    cost = np.squeeze(cost)

    return cost

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = ( A, W, b )

    return Z, cache

def linear_activation_forward(A, W, b, activation):
    if activation == 'relu':
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        Z, linear_cache = linear_forward(A, W, b)
        A, activation_cache = softmax(Z.T)

    cache = ( linear_cache, activation_cache )
    return A, cache

def L_model_forward(X, params):
    caches = []
    A = X
    # 因为 w1 与 b1 是成双成对的
    L = len(params) // 2

    # 一直到输出层
    for i in range(1, L):
        A, cache = linear_activation_forward(
            A,
            params["W" + str(i)],
            params["b" + str(i)],
            activation='relu'
        )
        caches.append(cache)
    
    # 输出层经 softmax 输出
    A, cache = linear_activation_forward(
        A,
        params["W" + str(L)],
        params["b" + str(L)],
        activation='softmax'
    )
    caches.append(cache)

    return A, caches

# dZ 为损失函数对该层的下一层输入的导数
def linear_backward(dZ, cache):
    A, W, b = cache
    # 
    m = A.shape[1]

    # 损失函数对该层到下一层权重的导数
    # (1 / m) 应该是因为有 m 个样本，需要求均值
    # 简单画俩矩阵就能算明白，比如 dZ 为 8000 * 10 ，A.T 为 50 * 8000
    # dz dot A.T 得到的矩阵的每一个值均为所有样本之和
    dW = (1 / m) * np.dot(dZ, A.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    # 最后求损失函数对该层输出的导数，为前一层的向后传播做准备
    dA = np.dot(W.T, dZ)

    return dA, dW, db

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'softmax':
        return
    return dA, dW, db

def L_model_backward(A, Y, caches):
    grads = {}
    # 除了输入层外其他层的个数
    L = len(caches)
    # 训练集样本个数
    m = A.shape[1]
    # 将 y 的 shape 变为与输出层输出的 shape 一样
    # 即 10 * 训练集样本数量。
    # 为了方便计算 softmax 的导数
    Y = Y.reshape(A.shape)
    
    # 在这里为什么不用 softmax 交叉熵导数 dA = A - Y
    # 直接来求呢。原因是为了代码统一。。

    # 直接求 softmax 交叉熵导数
    dZ = A - Y
    current_cache = caches[-1]
    # 直接求输出层的 dW 等
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, current_cache[0])

    for i in reversed(range(L - 1)):
        current_cache = caches[i]
        dA_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(i + 2)], current_cache, activation="relu")

        grads["dA" + str(i + 1)] = dA_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp

    return grads

def update_params(params, grads, alpha):
    L = len(params) // 2  # number of layers in the neural network

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - alpha * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - alpha * grads["db" + str(l + 1)]
    
    return params

# layers_dims：Array<Int> 包含了网络所有层的神经元数量
# Y 为向量形式的手写字符类别 et: [[1,0,0,0,0,0,0,0,0,0], xx, xx]
def model_DL( X, Y, Y_real, test_x, test_y, layers_dims, alpha, num_iterations, print_cost):  # lr was 0.009
    np.random.seed(2)
    costs = []

    # 初始化权重矩阵
    params = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
        # 前向传播
        A_last, caches = L_model_forward(X, params)
        # 计算交叉熵即误差
        cost = compute_cost(A_last, Y)
        # 反向传播
        grads = L_model_backward(A_last, Y, caches)

        if (i > 800 and i < 1700):
            alpha1 = 0.80 * alpha
            params = update_params(params, grads, alpha1)
        elif(i >= 1700):
            alpha1 = 0.50 * alpha
            params = update_params(params, grads, alpha1)
        else:
            params = update_params(params, grads, alpha)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
    
    predictions = predict(params, X)
    print("Train accuracy: {} %", sum(predictions == Y_real) / (float(len(Y_real))) * 100)
    predictions = predict(params, test_x)
    print("Test accuracy: {} %", sum(predictions == test_y) / (float(len(test_y))) * 100)

    return params

def predict(parameters, X):
    A_last, cache = L_model_forward(X, parameters)
    predictions = np.argmax(A_last, axis=0)
    return predictions