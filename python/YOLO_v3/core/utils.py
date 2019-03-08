import numpy as np

# 预训练模型是二进制文件，https://blog.csdn.net/haoqimao_hard/article/details/82109015
# 直接全局搜索该函数
def load_weights(var_list, weights_file):  
    with open(weights_file, "rb") as fp:
        # 跳过前5个int32值并以列表的形式读取其他内容
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        ptr = 0
        i = 0
        assign_ops = []
        while i < len(var_list) - 1:
            