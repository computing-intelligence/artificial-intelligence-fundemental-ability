import pickle 


with open('models/logistic_regression', 'rb') as f:
    model = pickle.load(f)
