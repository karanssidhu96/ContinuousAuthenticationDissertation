import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from numpy import reshape

NO_OF_FILES = 11

#For 5 second windows -> 162000 windows of 500 rows -> 1st half genuine & second half fraudulent
def create_training_window(training_data):
    windows = []
    for i in range(0,81000,500):
        window = []
        for j in range(9):
            window.append(training_data.iloc[i:i+500, j].values)
        window.append(1)
        windows.append(window)

    for i in range(81000,162000,500):
        window = []
        for j in range(9):
            window.append(training_data.iloc[i:i + 500, j].values)
        window.append(0)
        windows.append(window)
    return windows

#For 5 second windows => 54000 windows of 500 rows -> 1st half genuine & second half fraudulent
def create_testing_window(testing_data):
    windows = []
    for i in range(0,27000, 500):
        window = []
        for j in range(9):
            window.append(testing_data.iloc[i:i+500, j].values)
        window.append(1)
        windows.append(window)

    for i in range(27000,54000,500):
        window = []
        for j in range(9):
            window.append(testing_data.iloc[i:i + 500, j].values)
        window.append(0)
        windows.append(window)
    return windows

def load_data():
    print('Loading data')
    for i in range(10, 21):
        training_files.append('Supervised_100_Hz/train_activity_experiment_f100_acc_exp1_' + str(i) + '.csv')
        testing_files.append('Supervised_100_Hz/test_activity_experiment_f100_acc_exp1_' + str(i) + '.csv')

def seperate_data_from_labels(window, x, y):
    x_internal = []
    y_internal = []
    for i in range(len(window)):
        x_internal.append(window[i][:-1])
        y_internal.append(window[i][-1])
    x.append(x_internal)
    y.append(y_internal)

def scaler_training(scaler, train_file):
    train_file_scaled = scaler.fit_transform(train_file)
    train_file = pd.DataFrame(train_file_scaled)
    return train_file


def scaler_testing(scaler, test_file):
    test_file_scaled = scaler.transform(test_file)
    test_file = pd.DataFrame(test_file_scaled)
    return test_file

def train_user(classifier, x, y, no_of_windows, rows_per_window):
    x_input = reshape(x, [no_of_windows, rows_per_window])
    classifier.fit(x_input, y)

def test_user(classifier, x, no_of_windows, rows_per_window):
    x_input = reshape(x, [no_of_windows, rows_per_window])
    predictions = classifier.predict(x_input)
    return predictions

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
    linear_scaler_to_unit_variance = preprocessing.StandardScaler()
    load_data()

    print('Second Loop')
    for i in range(NO_OF_FILES):
        current_train_file = pd.read_csv(training_files[i])
        current_test_file = pd.read_csv(testing_files[i])

        #current_train_file = scaler_training(min_max_scaler, current_train_file)
        #current_test_file = scaler_testing(min_max_scaler, current_test_file)

        current_train_file = scaler_training(linear_scaler_to_unit_variance, current_train_file)
        current_test_file = scaler_testing(linear_scaler_to_unit_variance, current_test_file)

        training_data.append(current_train_file)
        testing_data.append(current_test_file)

        training_window = create_training_window(training_data[i])
        testing_window = create_testing_window(testing_data[i])

        seperate_data_from_labels(training_window, x, y)
        seperate_data_from_labels(testing_window, test_x, test_y)

    classifier = RandomForestClassifier(n_estimators=50, random_state=101)

    precision = 0
    for i in range(NO_OF_FILES):
        print('Training User ',i+1)
        train_user(classifier, x[i], y[i], 324, 4500)
        print('Testing User ',i+1)
        predictions = test_user(classifier, test_x[i], 108, 4500)

        precision = precision + precision_score(test_y[i], predictions)

    #confusion = confusion_matrix(actual, predictions)
    #print(confusion)
    #print(classification_report(actual, predictions))
    #print(precision_score(actual, predictions))
    precision = precision/NO_OF_FILES
    print(precision)