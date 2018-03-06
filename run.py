import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from numpy import reshape

def create_training_window(training_data):
    windows = []
    for i in range(0,81000,500):
        window = []
        for j in range(0,9):
            window.append(training_data.iloc[i:i+500, j].values)
        window.append(1)
        windows.append(window)

    for i in range(81000,162000,500):
        window = []
        for j in range(0,9):
            window.append(training_data.iloc[i:i + 500, j].values)
        window.append(0)
        windows.append(window)
    return windows

def create_testing_window(testing_data):
    windows = []
    for i in range(0,27000, 500):
        window = []
        for j in range(0,9):
            window.append(testing_data.iloc[i:i+500, j].values)
        window.append(1)
        windows.append(window)

    for i in range(27000,54000,500):
        window = []
        for j in range(0,9):
            window.append(testing_data.iloc[i:i + 500, j].values)
        window.append(0)
        windows.append(window)
    return windows

if __name__== '__main__':
    training_files = []
    testing_files = []
    training_data = []
    testing_data = []
    x = []
    y = []
    test_x = []
    test_y = []

    min_max_scaler = preprocessing.MinMaxScaler()

    print('First Loop')
    for i in range(10,21):
        training_files.append('Supervised_100_Hz/train_activity_experiment_f100_acc_exp1_' + str(i) +'.csv')
        testing_files.append('Supervised_100_Hz/test_activity_experiment_f100_acc_exp1_' + str(i) + '.csv')

    print('Second Loop')
    for i in range(11):
        current_train_file = pd.read_csv(training_files[i])
        current_test_file = pd.read_csv(testing_files[i])

        current_train_file_scaled = min_max_scaler.fit_transform(current_train_file)
        current_train_file = pd.DataFrame(current_train_file_scaled)

        current_test_file_scaled = min_max_scaler.transform(current_test_file)
        current_test_file = pd.DataFrame(current_test_file_scaled)

        training_data.append(current_train_file)
        testing_data.append(current_test_file)

        training_window = create_training_window(training_data[i])
        #print(len(training_window))
        #print(training_window[-1])
        testing_window = create_testing_window(testing_data[i])

        x_internal = []
        y_internal = []
        test_x_internal = []
        test_y_internal = []

        for windows in range(0, len(training_window)):
            x_internal.append(training_window[windows][:-1])
            y_internal.append(training_window[windows][-1])
        x.append(x_internal)
        #x = x + x_internal
        #concatenate((x,x_internal),axis=0)
        #print(x[0][0]) 324 per experiment
        y.append(y_internal)
        #y = y + y_internal
        #concatenate((y,y_internal),axis=0)

        for windows in range(0, len(testing_window)):
            test_x_internal.append(testing_window[windows][:-1])
            test_y_internal.append(testing_window[windows][-1])
        test_x.append(test_x_internal)
        #test_x = test_x + test_x_internal
        #concatenate((test_x,test_x_internal),axis=0)
        #print(len(test_x_internal)) 108 per experiment
        test_y.append(test_y_internal)
        #test_y = test_y + test_y_internal
        #concatenate((test_y, test_y_internal), axis=0)

    classifier = RandomForestClassifier(n_estimators=50, random_state=101)

    precision = 0
    for i in range(0, 11):
        print('Training User ',i+1)
        #print(len(x[i]))
        #print(x[i])
        #print(y[i])
        x_input = reshape(x[i], [324, 4500])
        classifier.fit(x_input,y[i])
        print('Testing User ',i+1)
        test_x_input = reshape(test_x[i], [108, 4500])
        predictions = classifier.predict(test_x_input)

        #print(test_y[i])
        #print(predictions)
        precision = precision + precision_score(test_y[i], predictions)
        #print(precision)

    #confusion = confusion_matrix(actual, predictions)
    #print(confusion)
    #print(classification_report(actual, predictions))
    #print(precision_score(actual, predictions))
    precision = precision/11
    print(precision)