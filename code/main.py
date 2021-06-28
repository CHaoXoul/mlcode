#from sklearn.datasets import load_boston
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error
#from sklearn.ensemble import RandomForestRegressor
#from deepforest import CascadeForestRegressor
#import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("train.csv")
    #print(df)
    strs = df['target'].value_counts()
    #print(strs)

    #value_map = dict((v, i) for i, v in enumerate(strs.index))
    value_map = {'Class_1': 0, 'Class_2': 1, 'Class_3': 2, 'Class_4': 3}
    #print(value_map)

    df = df.replace({'target': value_map})
    df = df.drop(columns=['id'])
    x_train = df.iloc[:, :-1]
    y_train = df['target']
    #print(x_train)

    df = pd.read_csv("test.csv")
    # df = df.drop(columns=['id'])
    x_test = df.iloc[:, 1:]  # keep the id column for output

    #from sklearn.ensemble import RandomForestClassifier
    from deepforest import CascadeForestClassifier
    #from sklearn.metrics import accuracy_score

    model = CascadeForestClassifier(n_jobs=12, n_estimators=6, n_trees=800)
    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_test.values)

    proba = model.predict_proba(x_test.values)
    #output = pd.DataFrame({'id': x_test.index, 'Class_1': proba[:, 0], 'Class_2': proba[:, 1], 'Class_3': proba[:, 2],'Class_4': proba[:, 3]})
    #output.to_csv('my_submission.csv', index=False)

    df = pd.read_csv("test.csv")
    output = pd.DataFrame({'id': df['id'], 'Class_1': proba[:, 0], 'Class_2': proba[:, 1], 'Class_3': proba[:, 2],'Class_4': proba[:, 3]})
    output.to_csv('my_submission.csv', index=False)

    #model = RandomForestClassifier(n_jobs=2, n_estimators=500)
    #model.fit(x_train, y_train)
    #proba = model.predict_proba(x_test)
    ## acc = accuracy_score(y_test, y_pred) * 100
    ## print("\nTesting Accuracy: {:.3f} %".format(acc))
    #output = pd.DataFrame({'id': df['id'], 'Class_1': proba[:, 0], 'Class_2': proba[:, 1], 'Class_3': proba[:, 2],'Class_4': proba[:, 3]})
    #output.to_csv('my_submission_rf.csv', index=False)