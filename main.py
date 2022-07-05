import numpy as np
import pandas as pd 
from utilities import split_data, sigma_from_median
from embeddings import seq_compress, seq_transform
from model import KernelLogisticRegression
np.random.seed(20)

def start():
    test_seq=input(
        'Please enter the path of the X_test sequences'
    )
    test_vec=input(
        'Please enter the path of the X_test vectors'
    )

    train_data = pd.read_csv('data/Xtr.csv')
    train_vectors = pd.read_csv('data/Xtr_vectors.csv')
    train_labels = pd.read_csv('data/Ytr.csv')
    test_df = pd.read_csv(test_vec)

    # Gettinng new embeddings
    dummy_df = train_data.copy()
    dummy_df = dummy_df.Sequence.apply(seq_compress)#, axis='columns')
    new_df = pd.DataFrame(dummy_df.to_list(), columns=range(64))
    df = pd.read_csv(test_seq)
    df = df.drop('Id', inplace=False, axis=1)
    df = df.Sequence.apply(seq_compress)#, axis='columns')

    # Split original data
    x_train, x_test, y_train, y_test = split_data(train_vectors, train_labels, p=0.5)
    x_train.drop('Id', inplace=True, axis=1)
    x_test.drop('Id', inplace=True, axis=1)
    y_train['Covid'] = 2*y_train['Covid'] - 1 # transform from {0, 1} to {-1, 1}
    y_test['Covid'] = 2*y_test['Covid'] - 1

    # Split new embeddings data

    x_train2, x_test2, y_train2, y_test2 = split_data(new_df, train_labels)
    y_train2['Covid'] = 2*y_train2['Covid'] - 1 # transform from {0, 1} to {-1, 1}
    y_test2['Covid'] = 2*y_test2['Covid'] - 1
   
    #sModel parameters
    s_median = sigma_from_median(x_train.to_numpy())
    
    kernel_parameters = {
        'degree': 3,
        'sigma': s_median,
    }
    training_parameters = {
        'fit_intercept': False,
        'lr': 0.01,
        'method': 'newton'
    }

    # Making predictions

    model = KernelLogisticRegression(lambd=0.001, kernel='rbf', **kernel_parameters)
    model.fit(x_train.to_numpy(), y_train.Covid.to_numpy(), **training_parameters)
    y_pred = model.predict(x_test.to_numpy())
    predictions_list = np.where(y_pred==-1, 0,1)
    predictions_df = pd.DataFrame({'Id': test_df.Id.to_numpy(), 'Covid': predictions_list})
    predictions_df.to_csv('submission_01.csv',index = False)

    testing_df = pd.DataFrame(df.to_list(), columns=range(64))
    model = KernelLogisticRegression(lambd=0.01, kernel='linear', **kernel_parameters)
    model.fit(x_train2.to_numpy(), y_train2.Covid.to_numpy(), **training_parameters)
    y_pred = model.predict(testing_df.to_numpy())
    predictions_list = np.where(y_pred==-1, 0,1)
    predictions_df = pd.DataFrame({'Id': test_df.Id.to_numpy(), 'Covid': predictions_list})
    predictions_df.to_csv('submission_02.csv',index = False)

    print('The submission for the logistic regression with standard embeddings is under submisson_01.csv')
    print('The submission for the logistic regression with new embeddings is under submisson_02.csv')

if __name__=="__main__":
    start()