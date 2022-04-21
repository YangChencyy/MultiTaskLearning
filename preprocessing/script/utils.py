import numpy as np
import os

save_path = "wav_emb/wav_emb_{}/"

def save_npy(vec, filename, size: int = 100):
    """save the numpy file with size specified.
    
    The numpy array vec will be automatically saved to path: "<save_path(size specified)>/<filename>"
    
    size is the number of rows of each audio embedding. In other words, each embedding should have the shape [size, ?]
    
    filename should be in the form of such as: "360_OpenSMILE2.3.0_mfcc.npy"
    """
    path = save_path.format(str(size))
    os.makedirs(save_path.format(str(size)), exist_ok = True)
    
    with open(path + filename, "wb") as f:
        np.save(f, vec)
    

def load_npy(filename, size: int = 100, raw: bool = False):
    """load the numpy file from the local file.
    
    if raw == True, it means you would like to load the raw data from path "wav_emb/"
        This is only used when you want to do the filtering for data preprocessing.
    
    size is the number of rows of each audio embedding.
    
    Otherwise, you should set raw = False, and load from the data you have filtered and saved.
    
    filename should be in the form of such as: "360_OpenSMILE2.3.0_mfcc.npy"
    """
    if raw:
        path = "wav_emb/"
    else:
        path = save_path.format(str(size))

    with open(path + filename, 'rb') as f:
        res = np.load(f)
    
    return res

def sample_data(vec, size: int = 100, step: int = 100, flatten: bool = True, seed: int = 0):
    """Randomly sample the data from raw audio embedding.
    
    step indicates the time length of the unit audio recording you want. 
    
    Return:
        res (numpy.ndarray): sampled embedding with shape [size, 39 * step] (flatten = True) or [size * step, 39] (flatten = False)
    """
    np.random.seed(seed)
    indices = np.random.normal(0, 1, size)
    indices_min = indices.min()
    indices_max = indices.max()
    indices = ((indices - indices_min) / (indices_max - indices_min)  * (vec.shape[0]-step)).astype(np.int32)

    if flatten:
        res = np.empty((size, vec.shape[1]*step))
        for i in range(size):
            print(vec[indices[i]:indices[i]+step].flatten().shape)
            res[i] = vec[indices[i]:indices[i]+step].flatten()
    else:
        res = np.empty((size*step, vec.shape[1]))
        for i in range(size):
            res[i*step:(i+1)*step] = vec[indices[i]:indices[i]+step]
    
    return res


if __name__ == "__main__":
    
    filename = "360_OpenSMILE2.3.0_mfcc.npy"
    vec = load_npy(filename, raw=True)
    # print(vec.shape)
    res = sample_data(vec)
    # print(res.shape)
    
    save_npy(res, filename)