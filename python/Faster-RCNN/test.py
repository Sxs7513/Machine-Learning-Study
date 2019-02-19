import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf                                                                                                                                                                        
import numpy as np                                                                                                         
                                                                                                                            
with tf.Session() as sess:                                                                                                 
    x=np.asarray([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)                                                                  
                                                                                                                            
    #  [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]                                                                           
    print(x)                                                                                                               
                                                                                                                            
    # keep_prob= 0.1，预期有9个数据*0, 实际上不一定                                                                        
    out = tf.nn.dropout(x, 0.5)                                                                                            
                                                                                                                            
    # 注意有可能是                            
    # 变为原来的10倍是因为里面有一个操作是 x / keep_prob                                                                               
    #  [ 0. 20. 30.  0.  0.  0.  0.  0.  0.  0.]                                                                           
    #  [ 0. 0. 30.  0.  0.  0.  0.  0.  0.  0.]                                                                            
    print(out.eval())  