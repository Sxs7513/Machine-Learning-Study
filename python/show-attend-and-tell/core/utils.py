import numpy as np
import cPickle as pickle
import hickle
import time
import os


def load_coco_data(data_path='./data', split='train'):
    data_path = os.path.join(data_path, split)
    start_t = time.time()
    data = {}

    data['features'] = hickle.load(os.path.join(data_path, '%s.features.hkl' %split))
    with open(os.path.join(data_path, '%s.file.names.pkl' %split), 'rb') as f:
        data['file_names'] = pickle.load(f)   
    with open(os.path.join(data_path, '%s.captions.pkl' %split), 'rb') as f:
        data['captions'] = pickle.load(f)
    with open(os.path.join(data_path, '%s.image.idxs.pkl' %split), 'rb') as f:
        data['image_idxs'] = pickle.load(f)
    
    for k, v in data.iteritems():
        if type(v) == np.ndarray:
            print (k, type(v), v.shape, v.dtype)
        else:
            print (k, type(v), len(v))
    end_t = time.time()
    print ("Elapse time: %.2f" %(end_t - start_t))
    return data


def load_pickle(path):
    with open(path, 'rb') as f:
        file = pickle.load(f)
        print ('Loaded %s..' %path)
        return file


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print ('Saved %s..' %path)