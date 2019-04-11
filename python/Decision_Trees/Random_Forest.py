# https://segmentfault.com/a/1190000013359859
# http://www.fdlly.com/p/207551054.html
# 随机树森林
import pandas as pd
import numpy as np


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
    
    n_folds = 5
    max_depth = 10
    min_size = 1
    sample_size = 1.0
    # 
    n_features = np.sqrt([data.shape[1]]).astype("int32")s
