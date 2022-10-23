import pickle


some_list = [i for i in range(100)]


with open('some_name', 'wb') as f:
    pickle.dump(some_list, f)





