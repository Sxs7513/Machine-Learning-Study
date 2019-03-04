import simpleflow as sf

with sf.Graph().as_default():
    w = sf.constant([[1, 2, 3], [3, 4, 5]], name='w')
    x = sf.constant([[9, 8], [7, 6], [10, 11]], name='x')
    b = sf.constant(1.0, 'b')
    result = sf.add(sf.matmul(w, x), b)
    
    with sf.Session() as sess:
        print(sess.run(result))