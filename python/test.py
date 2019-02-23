import tensorflow as tf

a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用

b = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)                                  # 判断GPU是否可以用

print(a)
print(b)
