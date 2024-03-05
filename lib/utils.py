import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def swarmplot_subjects(res, key, sort=True):
    '''
    docstring here
    '''
    
    data_toplot = []
    sub_toplot = []
    
    for sub in res['subject'].unique():
        tmp1 = res[res['subject']==sub]
        for sess in tmp1['target'].unique():
            tmp2 = tmp1[tmp1['target']==sess]
            data_toplot.append(np.array(tmp2[key]))
            sub_toplot.append(sub)
    if sort:
        # Sort by the mean value
        sort_idx = np.argsort([np.mean(d) for d in data_toplot])
        data_sorted = []; sub_sorted = []
        for i in sort_idx:
            data_sorted.append(data_toplot[i])
            sub_sorted.append(sub_toplot[i])
    else:
        data_sorted = data_toplot
        sub_sorted = sub_toplot
                        
    return data_sorted, sub_sorted

def construct_distance_adjacency(coords):
    '''
    Takes a Nx3 matrix of electrode coordinates and constructs a Euclidean distance matrix. 
    '''    
    adj = np.empty([coords.shape[0], coords.shape[0]])
    
    for i in range(coords.shape[0]):
        for j in range(coords.shape[0]):
            adj[i, j] = np.linalg.norm(coords[i, :]-coords[j, :])
    
    return adj