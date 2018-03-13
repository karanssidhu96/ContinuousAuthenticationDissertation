import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing, svm
from numpy import reshape
from timeit import default_timer as timer

NO_OF_FILES = 65
NO_OF_FEATURES = 9
GENUINE_LABEL = 1
FRAUDULENT_LABEL = 0
NO_ROWS_TRAINING_100_HZ = 162000
NO_ROWS_TESTING_100_HZ = 54000
ROWS_PER_WINDOW_100_HZ_5_SECONDS = 500
ROWS_PER_WINDOW_100_HZ_1_SECOND = 100
NO_ROWS_TRAINING_25_HZ = 40500
NO_ROWS_TESTING_25_HZ = 13500
ROWS_PER_WINDOW_25_HZ_5_SECONDS = 125
ROWS_PER_WINDOW_25_HZ_1_SECOND = 25


#For 5 second windows -> 162000 rows total, 500 rows per window -> 1st half genuine & second half fraudulent
def create_window(data, no_rows_per_file, no_rows_per_window):
    windows = []
    for i in range(0,int(no_rows_per_file/2),no_rows_per_window):
        window = []
        for j in range(NO_OF_FEATURES):
            window.append(data.iloc[i:i + no_rows_per_window, j].values)
        window.append(GENUINE_LABEL)
        windows.append(window)

    for i in range(int(no_rows_per_file/2), no_rows_per_file,no_rows_per_window):
        window = []
        for j in range(NO_OF_FEATURES):
            window.append(data.iloc[i:i + no_rows_per_window, j].values)
        window.append(FRAUDULENT_LABEL)
        windows.append(window)
    return windows


def load_experiment(start, end, experiment, frequency):
    for i in range(start, end):
        training_files.append('Supervised_'+ str(frequency) +'_Hz/train_activity_experiment_f'+ str(frequency) +'_acc_exp' + str(experiment) + '_' + str(i) + '.csv')
        testing_files.append('Supervised_'+ str(frequency) +'_Hz/test_activity_experiment_f'+ str(frequency) +'_acc_exp' + str(experiment) + '_' + str(i) + '.csv')

def load_experiment_25_Hz(start, end, experiment):
    for i in range(start, end):
        training_files.append('Supervised_25_Hz/train_activity_experiment_f25_acc_exp' + str(experiment) + '_' + str(i) + '.csv')
        testing_files.append('Supervised_25_Hz/test_activity_experiment_f25_acc_exp' + str(experiment) + '_' + str(i) + '.csv')

def load_data(frequency):
    print('Loading data')
    #start = 10
    #end = 21
    #for experiment in range(1,7):
        #load_experiment(start, end, experiment)
        #start = start + 10
        #end = end + 10
    load_experiment(10, 21, 1, frequency)
    load_experiment(20, 31, 2, frequency)
    load_experiment(30, 40, 3, frequency)
    load_experiment(40, 51, 4, frequency)
    load_experiment(50, 61, 5, frequency)
    load_experiment(60, 71, 6, frequency)

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
    training_files, testing_files, training_data, testing_data, x, y, test_x, test_y = [], [], [], [], [], [], [], []

    min_max_scaler = preprocessing.MinMaxScaler()
    linear_scaler_to_unit_variance = preprocessing.StandardScaler()
    # 100 Hz
    load_data(100)
    # 25 Hz
    #load_data(25)

    print('Second Loop')
    for i in range(NO_OF_FILES):
        current_train_file = pd.read_csv(training_files[i])
        current_test_file = pd.read_csv(testing_files[i])
        print('Scaling and windowing user:', i)
        #Fails on experiment 3 34 both frequencies

        #current_train_file = scaler_training(min_max_scaler, current_train_file)
        #current_test_file = scaler_testing(min_max_scaler, current_test_file)

        current_train_file = scaler_training(linear_scaler_to_unit_variance, current_train_file)
        current_test_file = scaler_testing(linear_scaler_to_unit_variance, current_test_file)

        training_data.append(current_train_file)
        testing_data.append(current_test_file)

        # 5 second window 100 Hz
        training_window = create_window(training_data[i], NO_ROWS_TRAINING_100_HZ, ROWS_PER_WINDOW_100_HZ_5_SECONDS)
        testing_window = create_window(testing_data[i], NO_ROWS_TESTING_100_HZ, ROWS_PER_WINDOW_100_HZ_5_SECONDS)
        # 5 second window 25 Hz
        #training_window = create_window(training_data[i], NO_ROWS_TRAINING_25_HZ, ROWS_PER_WINDOW_25_HZ_5_SECONDS)
        #testing_window = create_window(testing_data[i], NO_ROWS_TESTING_25_HZ, ROWS_PER_WINDOW_25_HZ_5_SECONDS)
        # 1 second window 100 Hz
        #training_window = create_window(training_data[i], NO_ROWS_TRAINING_100_HZ, ROWS_PER_WINDOW_100_HZ_1_SECOND)
        #testing_window = create_window(testing_data[i], NO_ROWS_TESTING_100_HZ, ROWS_PER_WINDOW_100_HZ_1_SECOND)
        # 1 Second Window 25 Hz
        #training_window = create_window(training_data[i], NO_ROWS_TRAINING_25_HZ, ROWS_PER_WINDOW_25_HZ_1_SECOND)
        #testing_window = create_window(testing_data[i], NO_ROWS_TESTING_25_HZ, ROWS_PER_WINDOW_25_HZ_1_SECOND)

        seperate_data_from_labels(training_window, x, y)
        seperate_data_from_labels(testing_window, test_x, test_y)

    #classifier = RandomForestClassifier(n_estimators=50, random_state=101)
    classifier = svm.SVC(kernel='rbf', gamma=0.0001, random_state=101)

    precision, avg_train_time, avg_test_time = 0, 0, 0
    for i in range(NO_OF_FILES):
        print('Training User',i+1)
        start_time = timer()
        #5 Seconds Training 100 Hz
        train_user(classifier, x[i], y[i], 324, 4500)
        # 5 Seconds Training 25 Hz
        #train_user(classifier, x[i], y[i], 324, 1125)
        #1 Second Training 100 Hz
        #train_user(classifier, x[i], y[i], 1620, 900)
        #1 Second Training 25Hz
        #train_user(classifier, x[i], y[i], 1620, 225)
        end_time = timer()
        training_time = end_time - start_time
        print('Time to train user',i+1, ':', training_time)
        print('Testing User ',i+1)
        start_time = timer()
        # 5 Seconds Testing 100 Hz
        predictions = test_user(classifier, test_x[i], 108, 4500)
        # 5 Seconds Testing 25
        #predictions = test_user(classifier, test_x[i], 108, 1125)
        # 1 Second Testing 100 Hz
        #predictions = test_user(classifier, test_x[i], 540, 900)
        # 1 Second Testing 25 Hz
        #predictions = test_user(classifier, test_x[i], 540, 225)
        end_time = timer()
        testing_time = end_time - start_time
        print('Time to test user', i + 1, ':', testing_time)

        precision = precision + precision_score(test_y[i], predictions)
        avg_train_time = avg_train_time + training_time
        avg_test_time = avg_test_time + testing_time

    #confusion = confusion_matrix(actual, predictions)
    #print(confusion)
    #print(classification_report(actual, predictions))
    #print(precision_score(actual, predictions))
    precision = precision / NO_OF_FILES
    avg_train_time = avg_train_time / NO_OF_FILES
    avg_test_time = avg_test_time / NO_OF_FILES
    print('Accuracy:', precision, "\nAverage Training Time Per User:", avg_train_time, "\nAverage Testing Time Per User:", avg_test_time)