import numpy as np

def sample_stratified(n, data, idx, interval):
    """
        Sample n samples from each class by the interval
    Input:    
        n : int, number of samples
        data: include 'data' and 'labels'
        idx: index of indicator to be stratified
        interval: if set to [1.01, 1.2], then the interval is (-10000, 1.01], (1.01, 1.2], (1.2, 10000)
    Output:
        idx0, idx1
    """

    land = np.logical_and
    lor = np.logical_or

    def worker(li):
        idx_li, l, s = li
        return np.array(idx_li)[np.random.choice(l, s, replace=False)]

    x = data['data'][:, idx]
    y = data['labels']
    index = np.array(range(x.__len__()))
    
    process = []
    
    for i in range(2):
        count = 0
        mi = y==i
        xi = x[mi]
        indexi = index[mi]
        r = float(n) / len(xi) # Sampling rate
        left = -10000
        for right in interval:
            m = land(xi > left, xi < right+0.0001)
            q = m.sum()
            p = int(r*q+0.5)
            process.append([indexi[m], q, p])
            left = right
            count += p
        
        right = 10000
        m = land(xi > left, xi < right+0.0001)
        q = m.sum()
        process.append([indexi[m], q, n-count])
       
    process0 = list(map(process.pop, [0]*(len(process)//2)))  # The first half is class 0 and the second half is class 1
    process1 = process
    
    idx0 = list(map(worker, process0)); idx0 = [j for i in idx0 for j in i]
    idx1 = list(map(worker, process1)); idx1 = [j for i in idx1 for j in i]

    return idx0, idx1


if __name__ == "__main__":
    
    data = {'data':np.random.rand(50, 3), 'labels':np.random.randint(0, 2, 50)}
    sample_stratified(10, data, 0, [0.1, 0.2, 0.3])