import os
import csv
import sys
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from Config import Config


def get_feature_opensmile(filepath: str):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    cmd = 'cd ' + Config.OPENSMILE_PATH + ' && ./SMILExtract -C config/' + Config.CONFIG + '.conf -I ' + filepath + ' -O ' + BASE_DIR + '/' + Config.FEATURE_PATH + 'single_feature.csv'
    print("Opensmile cmd: ", cmd)
    os.system(cmd)
    
    reader = csv.reader(open(BASE_DIR + '/' + Config.FEATURE_PATH + 'single_feature.csv','r'))
    rows = [row for row in reader]
    last_line = rows[-1]
    return last_line[1: Config.FEATURE_NUM[Config.CONFIG] + 1]



def load_feature(feature_path: str, train: bool):
    df = pd.read_csv(feature_path)
    features = [str(i) for i in range(1, Config.FEATURE_NUM[Config.CONFIG] + 1)]

    X = df.loc[:,features].values
    Y = df.loc[:,'label'].values

    if train == True:
        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, Config.MODEL_PATH + 'SCALER_OPENSMILE.m')
        X = scaler.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        return x_train, x_test, y_train, y_test
    else:
        scaler = joblib.load(Config.MODEL_PATH + 'SCALER_OPENSMILE.m')
        X = scaler.transform(X)
        return X


def get_data(data_path: str, feature_path: str, train: bool):

    writer = csv.writer(open(feature_path, 'w'))
    first_row = ['label']
    for i in range(1, Config.FEATURE_NUM[Config.CONFIG] + 1):
        first_row.append(str(i))
    writer.writerow(first_row)

    writer = csv.writer(open(feature_path, 'a+'))
    print('Opensmile extracting...')

    if train == True:
        cur_dir = os.getcwd()
        sys.stderr.write('Curdir: %s\n' % cur_dir)
        os.chdir(data_path)
        for i, directory in enumerate(Config.CLASS_LABELS):
            sys.stderr.write("Started reading folder %s\n" % directory)
            os.chdir(directory)

            # label_name = directory
            label = Config.CLASS_LABELS.index(directory)

            for filename in os.listdir('.'):
                if not filename.endswith('wav'):
                    continue
                filepath = os.getcwd() + '/' + filename
                
                feature_vector = get_feature_opensmile(filepath)
                feature_vector.insert(0, label)
                writer.writerow(feature_vector)

            sys.stderr.write("Ended reading folder %s\n" % directory)
            os.chdir('..')
        os.chdir(cur_dir)
    
    else:
        feature_vector = get_feature_opensmile(data_path)
        feature_vector.insert(0, '-1')
        writer.writerow(feature_vector)

    print('Opensmile extract done.')

    if(train == True):
        return load_feature(feature_path, train = train)
