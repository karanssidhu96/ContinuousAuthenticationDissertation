import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, svm
from sklearn.neural_network import MLPClassifier
from numpy import reshape, mean, std, var, cov, abs, power, sqrt
from timeit import default_timer as timer
from scipy.stats import iqr, skew, kurtosis
from scipy.fftpack import fft

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
HZ = 100
WINDOW = 5
CLASSIFIER = 'Random Forest'
SCALING = 'Standardization'
#Normalization
FEATURE_ENGINEERING = True
CORRELATIONS = True
LENGTHS_ANGLES = True


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

def create_window_FE(data, no_rows_per_file, no_rows_per_window):
    windows = []
    for i in range(0,int(no_rows_per_file/2),no_rows_per_window):
        window, means, SDs, accelerometer, gyroscope, magnetometer = [], [], [], [], [], []
        for j in range(NO_OF_FEATURES):
            if LENGTHS_ANGLES:
                if j < 3:
                    accelerometer.extend(data.iloc[i:i + no_rows_per_window, j].values)
                elif 3 <= j < 6:
                    gyroscope.extend(data.iloc[i:i + no_rows_per_window, j].values)
                else:
                    magnetometer.extend(data.iloc[i:i + no_rows_per_window, j].values)
            feature_raw = data.iloc[i:i + no_rows_per_window, j].values
            window.extend(fe_stats(feature_raw))
            means.append(mean(feature_raw))
            SDs.append(std(feature_raw))
            if j != 0 and j % 3 == 0 and CORRELATIONS or j == 8 and CORRELATIONS:
                window.append(correlation(means[0], means[1], SDs[0], SDs[1]))
                window.append(correlation(means[0], means[2], SDs[0], SDs[2]))
                window.append(correlation(means[1], means[2], SDs[1], SDs[2]))
                means, SDs = [], []
                means.append(mean(feature_raw))
                SDs.append(std(feature_raw))
        if LENGTHS_ANGLES:
            accelerometer_lengths = calculate_lengths_of_vectors(accelerometer, int(no_rows_per_window))
            gyroscope_lengths = calculate_lengths_of_vectors(gyroscope, int(no_rows_per_window))
            magnetometer_lengths = calculate_lengths_of_vectors(magnetometer, int(no_rows_per_window))
            accelerometer_angles = calculate_angles(accelerometer, accelerometer_lengths, int(no_rows_per_window))
            gyroscope_angles = calculate_angles(gyroscope, gyroscope_lengths, int(no_rows_per_window))
            magnetometer_angles = calculate_angles(magnetometer, magnetometer_lengths, int(no_rows_per_window))
            window.append(mean(accelerometer_lengths))
            window.append(mean(gyroscope_lengths))
            window.append(mean(magnetometer_lengths))
            for j in range(3):
                window.append(mean(accelerometer_angles[j]))
                window.append(mean(gyroscope_angles[j]))
                window.append(mean(magnetometer_angles[j]))
        window.append(GENUINE_LABEL)
        windows.append(window)

    for i in range(int(no_rows_per_file/2), no_rows_per_file,no_rows_per_window):
        window, means, SDs, accelerometer, gyroscope, magnetometer = [], [], [], [], [], []
        for j in range(NO_OF_FEATURES):
            if LENGTHS_ANGLES:
                if j < 3:
                    accelerometer.extend(data.iloc[i:i + no_rows_per_window, j].values)
                elif 3 <= j < 6:
                    gyroscope.extend(data.iloc[i:i + no_rows_per_window, j].values)
                else:
                    magnetometer.extend(data.iloc[i:i + no_rows_per_window, j].values)
            feature_raw = data.iloc[i:i + no_rows_per_window, j].values
            window.extend(fe_stats(feature_raw))
            means.append(mean(feature_raw))
            SDs.append(std(feature_raw))
            if j != 0 and j % 3 == 0 and CORRELATIONS or j == 8 and CORRELATIONS:
                window.append(correlation(means[0], means[1], SDs[0], SDs[1]))
                window.append(correlation(means[0], means[2], SDs[0], SDs[2]))
                window.append(correlation(means[1], means[2], SDs[1], SDs[2]))
                means, SDs = [], []
                means.append(mean(feature_raw))
                SDs.append(std(feature_raw))
        if LENGTHS_ANGLES:
            accelerometer_lengths = calculate_lengths_of_vectors(accelerometer, int(no_rows_per_window))
            gyroscope_lengths = calculate_lengths_of_vectors(gyroscope, int(no_rows_per_window))
            magnetometer_lengths = calculate_lengths_of_vectors(magnetometer, int(no_rows_per_window))
            accelerometer_angles = calculate_angles(accelerometer, accelerometer_lengths, int(no_rows_per_window))
            gyroscope_angles = calculate_angles(gyroscope, gyroscope_lengths, int(no_rows_per_window))
            magnetometer_angles = calculate_angles(magnetometer, magnetometer_lengths, int(no_rows_per_window))
            window.append(mean(accelerometer_lengths))
            window.append(mean(gyroscope_lengths))
            window.append(mean(magnetometer_lengths))
            for j in range(3):
                window.append(mean(accelerometer_angles[j]))
                window.append(mean(gyroscope_angles[j]))
                window.append(mean(magnetometer_angles[j]))
        window.append(FRAUDULENT_LABEL)
        windows.append(window)
    return windows

def fe_stats(feature):
    feature_stats = []
    feature_stats.append(max(feature))
    feature_stats.append(min(feature))
    feature_stats.append(mean(feature))
    feature_stats.append(std(feature))
    feature_stats.append(iqr(feature))
    feature_stats.append(var(feature))
    feature_stats.append(skew(feature))
    feature_stats.append(kurtosis(feature))
    feature_stats.append(calculate_energy(feature))
    return feature_stats

def correlation(X, Y, sd_x, sd_y):
    input_cov = [X, Y]
    if (sd_x * sd_y) != 0:
        return cov(input_cov)/(sd_x * sd_y)
    else:
        print('Can not calculate correlation')
        return 0

def calculate_energy(feature):
    energy = 0
    dft_values = fft(feature)
    for i in range(0,len(feature)):
        energy = energy + power(abs(dft_values[i]), 2)
    energy = energy/len(feature)
    return energy

def calculate_lengths_of_vectors(instrument_measures, no_of_rows):
    lengths = []
    for i in range(0, no_of_rows):
        x = instrument_measures[i]
        y = instrument_measures[i + no_of_rows]
        z = instrument_measures[i + (no_of_rows * 2)]
        lengths.append(calculate_vector_length(x, y, z))
    return lengths

def calculate_vector_length(x, y, z):
    return sqrt(power(x, 2) + power(y, 2) + power(z, 2))

def calculate_angles(instrument_measures, lengths, no_of_rows):
    x_angles = []
    y_angles = []
    z_angles = []
    for i in range(no_of_rows):
        x_angles.append(instrument_measures[i]/lengths[i])
        y_angles.append(instrument_measures[i + no_of_rows]/lengths[i])
        z_angles.append(instrument_measures[i + no_of_rows*2]/lengths[i])
    return [x_angles, y_angles, z_angles]


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
    load_experiment(10, 21, 1, frequency)
    load_experiment(20, 31, 2, frequency)
    load_experiment(30, 40, 3, frequency)
    # Fails on experiment 3 34 both frequencies
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

def normalization_linear_scaler(current_train_file, current_test_file):
    if SCALING == 'Normalization':
        training_data.append(scaler_training(min_max_scaler, current_train_file))
        testing_data.append(scaler_testing(min_max_scaler, current_test_file))
    else:
        training_data.append(scaler_training(linear_scaler_to_unit_variance, current_train_file))
        testing_data.append(scaler_testing(linear_scaler_to_unit_variance, current_test_file))

if __name__== '__main__':
    training_files, testing_files, training_data, testing_data, x, y, test_x, test_y = [], [], [], [], [], [], [], []

    min_max_scaler = preprocessing.MinMaxScaler()
    linear_scaler_to_unit_variance = preprocessing.StandardScaler()

    load_data(HZ)

    print('Second Loop')
    for i in range(NO_OF_FILES):
        current_train_file = pd.read_csv(training_files[i])
        current_test_file = pd.read_csv(testing_files[i])
        print('Scaling and windowing user:', i)

        if FEATURE_ENGINEERING:
            if HZ == 100:
                if WINDOW == 5:
                    # 5 second window 100 Hz
                    training_window = create_window_FE(current_train_file, NO_ROWS_TRAINING_100_HZ,
                                                    ROWS_PER_WINDOW_100_HZ_5_SECONDS)
                    testing_window = create_window_FE(current_test_file, NO_ROWS_TESTING_100_HZ,
                                                   ROWS_PER_WINDOW_100_HZ_5_SECONDS)
                else:
                    # 1 second window 100 Hz
                    training_window = create_window_FE(current_train_file, NO_ROWS_TRAINING_100_HZ,
                                                    ROWS_PER_WINDOW_100_HZ_1_SECOND)
                    testing_window = create_window_FE(current_test_file, NO_ROWS_TESTING_100_HZ,
                                                   ROWS_PER_WINDOW_100_HZ_1_SECOND)

            else:
                if WINDOW == 5:
                    # 5 second window 25 Hz
                    training_window = create_window_FE(current_train_file, NO_ROWS_TRAINING_25_HZ,
                                                    ROWS_PER_WINDOW_25_HZ_5_SECONDS)
                    testing_window = create_window_FE(current_test_file, NO_ROWS_TESTING_25_HZ,
                                                   ROWS_PER_WINDOW_25_HZ_5_SECONDS)
                else:
                    # 1 Second Window 25 Hz
                    training_window = create_window_FE(current_train_file, NO_ROWS_TRAINING_25_HZ,
                                                    ROWS_PER_WINDOW_25_HZ_1_SECOND)
                    testing_window = create_window_FE(current_test_file, NO_ROWS_TESTING_25_HZ,
                                                   ROWS_PER_WINDOW_25_HZ_1_SECOND)

            normalization_linear_scaler(training_window, testing_window)

        else:

            normalization_linear_scaler(current_train_file, current_test_file)

            if HZ == 100:
                if WINDOW == 5:
                    # 5 second window 100 Hz
                    training_window = create_window(training_data[i], NO_ROWS_TRAINING_100_HZ,
                                                    ROWS_PER_WINDOW_100_HZ_5_SECONDS)
                    testing_window = create_window(testing_data[i], NO_ROWS_TESTING_100_HZ,
                                                   ROWS_PER_WINDOW_100_HZ_5_SECONDS)
                else:
                    # 1 second window 100 Hz
                    training_window = create_window(training_data[i], NO_ROWS_TRAINING_100_HZ,
                                                    ROWS_PER_WINDOW_100_HZ_1_SECOND)
                    testing_window = create_window(testing_data[i], NO_ROWS_TESTING_100_HZ, ROWS_PER_WINDOW_100_HZ_1_SECOND)

            else:
                if WINDOW == 5:
                    # 5 second window 25 Hz
                    training_window = create_window(training_data[i], NO_ROWS_TRAINING_25_HZ,
                                                    ROWS_PER_WINDOW_25_HZ_5_SECONDS)
                    testing_window = create_window(testing_data[i], NO_ROWS_TESTING_25_HZ, ROWS_PER_WINDOW_25_HZ_5_SECONDS)
                else:
                    # 1 Second Window 25 Hz
                    training_window = create_window(training_data[i], NO_ROWS_TRAINING_25_HZ,
                                                    ROWS_PER_WINDOW_25_HZ_1_SECOND)
                    testing_window = create_window(testing_data[i], NO_ROWS_TESTING_25_HZ, ROWS_PER_WINDOW_25_HZ_1_SECOND)

        seperate_data_from_labels(training_window, x, y)
        seperate_data_from_labels(testing_window, test_x, test_y)

        if CLASSIFIER == 'Random Forest':
            classifier = RandomForestClassifier(n_estimators=50, random_state=101)
        elif CLASSIFIER == 'SVM':
            classifier = svm.SVC(kernel='rbf', C=100, gamma=100, random_state=101)
        elif CLASSIFIER == 'Logistic Regression':
            classifier = LogisticRegression(C=100, random_state=101)
        else:
            classifier = MLPClassifier(hidden_layer_sizes=(72,36), activation='logistic', random_state=101, alpha=0.01, max_iter=400)

    precision, avg_train_time, avg_test_time = 0, 0, 0
    for i in range(NO_OF_FILES):
        print('Training User',i+1)
        start_time = timer()
        if FEATURE_ENGINEERING:
            if CORRELATIONS:
                if LENGTHS_ANGLES:
                    if WINDOW == 5:
                        train_user(classifier, x[i], y[i], 324, 198)
                    else:
                        test_user(classifier, x[i], y[i], 1620, 198)
                else:
                    if WINDOW == 5:
                        train_user(classifier, x[i], y[i], 324, 90)
                    else:
                        train_user(classifier, x[i], y[i], 1620, 90)
            else:
                if WINDOW == 5:
                    train_user(classifier, x[i], y[i], 324, 81)
                else:
                    train_user(classifier, x[i], y[i], 1620, 81)
        elif HZ == 100:
            if WINDOW == 5:
                #5 Seconds Training 100 Hz
                train_user(classifier, x[i], y[i], 324, 4500)
            else:
                # 1 Second Training 100 Hz
                train_user(classifier, x[i], y[i], 1620, 900)
        else:
            if WINDOW == 5:
                # 5 Seconds Training 25 Hz
                train_user(classifier, x[i], y[i], 324, 1125)
            else:
                #1 Second Training 25Hz
                train_user(classifier, x[i], y[i], 1620, 225)
        end_time = timer()
        training_time = end_time - start_time
        print('Time to train user',i+1, ':', training_time)
        print('Testing User ',i+1)
        start_time = timer()
        if FEATURE_ENGINEERING:
            if CORRELATIONS:
                if LENGTHS_ANGLES:
                    if WINDOW == 5:
                        predictions = test_user(classifier, test_x[i], 108, 198)
                    else:
                        predictions = test_user(classifier, test_x[i], 540, 198)
                else:
                    if WINDOW == 5:
                        predictions = test_user(classifier, test_x[i], 108, 90)
                    else:
                        predictions = test_user(classifier, test_x[i], 540, 90)
            else:
                if WINDOW == 5:
                    predictions = test_user(classifier, test_x[i], 108, 81)
                else:
                    predictions = test_user(classifier, test_x[i], 540, 81)
        elif HZ == 100:
            if WINDOW == 5:
                # 5 Seconds Testing 100 Hz
                predictions = test_user(classifier, test_x[i], 108, 4500)
            else:
                # 1 Second Testing 100 Hz
                predictions = test_user(classifier, test_x[i], 540, 900)
        else:
            if WINDOW == 5:
                # 5 Seconds Testing 25
                predictions = test_user(classifier, test_x[i], 108, 1125)
            else:
                # 1 Second Testing 25 Hz
                predictions = test_user(classifier, test_x[i], 540, 225)
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