import numpy as np
np.random.seed(20)


def split_data(X,y, p=0.8):
  # first we will shuffle the data
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  X = X.iloc[indices]
  y = y.iloc[indices]

  #Next create a splitter to slice the data
  split = int(len(X)*p)
  x_train = X.iloc[0:split]
  x_test = X.iloc[split:]
  y_train = y.iloc[0:split]
  y_test = y.iloc[split:]
  return x_train, x_test, y_train, y_test

def sigma_from_median(X):
    '''
    Returns the median of ||Xi-Xj||
    
    Input
    -----
    X: (n, p) matrix
    '''
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)

def sigma_from_quantile(X, q):
   
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.quantile(euclidean_dist, q)

def accuracy(y, y_hat):
  count = 0
  for idx in range(len(y)):
    if y[idx]==y_hat[idx]:
      count +=1
  return count*100/len(y)

