import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple
from collections import OrderedDict
import os
sns.set(color_codes=True)


# fixme: add sorting to avoid potential unsorted ts mistakes? could impact epochs
def read_sensor_data_by_subject(base_path: str, cur_activity: str, cur_sub: str) -> Tuple[pd.DataFrame, List[str], str]:
    """
    :param base_path: data base path
    :param cur_activity: activity designation and number
    :param cur_sub: subject designation
    :return: a dataframe where all three sensor sources are merged (inner). This (for the sake of simplicity) retains
    observations that exist on all three sensors.
    The dataframe also includes activity and subject as meta-data.
    We also construct a label column using cur_activity
    In addition, this function also returns a list of features, which will later on help differentiate between features, meta-data, and label
    """
    sensors_dir_names = ['A_DeviceMotion_data', 'B_Accelerometer_data', 'C_Gyroscope_data']

    sensor_data_paths = [f"{base_path}{sensor}/{cur_activity}/{cur_sub}.csv" for sensor in sensors_dir_names]

    # Read motion sensor data
    sub_motion_df = pd.read_csv(sensor_data_paths[0])
    del sub_motion_df[sub_motion_df.columns[0]]

    # renaming acc and gyro feature cols to avoid conflict
    # read acc
    sub_acc_df = pd.read_csv(sensor_data_paths[1])
    del sub_acc_df[sub_acc_df.columns[0]]
    sub_acc_df.rename(columns={'x': 'x_acc', 'y': 'y_acc', 'z': 'z_acc'}, inplace=True)

    # read gyro
    sub_gyro_df = pd.read_csv(sensor_data_paths[2])
    del sub_gyro_df[sub_gyro_df.columns[0]]
    sub_gyro_df.rename(columns={'x': 'x_gyro', 'y': 'y_gyro', 'z': 'z_gyro'}, inplace=True)

    # Joining all subject's sensor data
    '''
    # Note: Different sensors have (slightly) different number of observations, which is totally normal 
    # for sensors data. For simplicity's sake I used only observations that exists across all three sensors
    # and threw out any excess information one of the sensors have.
    # In sampling rate of 50Hz, this usually resulted in losing less than 1 sec of data
    '''
    cur_sub_sensors_data_df = pd.concat([sub_motion_df, sub_acc_df, sub_gyro_df], axis=1, join='inner')
    features_cols = list(cur_sub_sensors_data_df)

    label_col = 'label'
    cur_sub_sensors_data_df[label_col] = cur_activity.split('_')[0]
    cur_sub_sensors_data_df['subject'] = cur_sub
    cur_sub_sensors_data_df['activity'] = cur_activity
    cur_sub_sensors_data_df['session_uid'] = f"{cur_sub}_{cur_activity}"  # session unique identifier
    cur_sub_sensors_data_df['ts'] = range(0, len(cur_sub_sensors_data_df))
    return cur_sub_sensors_data_df, features_cols, label_col


def read_all_subjects_activity_data_to_df(base_path: str) -> Tuple[pd.DataFrame, List[str], str]:
    """
    :param base_path: dataset base-path. Hard-coded assumption that directory structure is the same as provided in
    the original dataset:
        https://github.com/mmalekzadeh/motion-sense/
    :return: dataframe containing feature, meta-data and label. It also returns columns names lists: features, and label column name
    """
    activity_sets_list = ['dws_1', 'dws_2', 'dws_11', 'jog_9', 'jog_16', 'sit_5', 'sit_13', 'std_6', 'std_14', 'ups_3', 'ups_4', 'ups_12', 'wlk_7', 'wlk_8', 'wlk_15']
    subs_list = ['sub_' + str(i) for i in range(1, 25)]
    all_subject_sensor_data_list = []

    features_cols = label_col = None
    for action in activity_sets_list:
        print(f"Analysing action: {action}")
        for sub in subs_list:
            print(f"\r\tSubject: {sub}", end="")
            # subject_sensor_data_df, features_cols, meta_data_cols_list, label_col = read_sensor_data_by_subject(base_path, action, sub)
            subject_sensor_data_df, features_cols, label_col = read_sensor_data_by_subject(base_path, action, sub)
            all_subject_sensor_data_list.append(subject_sensor_data_df)
        print("")
    print("Done")
    print("Joining all users and sensors data... ")
    all_subject_sensor_data_df = pd.concat(all_subject_sensor_data_list, sort=False, ignore_index=True)
    print("Done")
    return all_subject_sensor_data_df, features_cols, label_col


def add_session_epochs(df: pd.DataFrame, session_uid: pd.Series, sampling_rate_hz: int = 50, seconds_per_epoch: int = 10):
    """
    :param df: input dataframe
    :param session_uid: session unique identifier to help locate samples belonging to the same
    :param sampling_rate_hz: sensors sampling rate. It is provided as 50Hz in the dataset description
    :param seconds_per_epoch: How many seconds to include per epoch
    :return: This function divides sessions into epochs. We do that since the number of users X activities is rather low.
    By subdividing the session, we may increase the number of observations per each action.
    This will make it easier to 
    """
    assert len(df) == len(session_uid), f"Dataframe and session must have the same length, but it isn't ({len(df)}, {len(session_uid)}). Cowardly aborting."
    assert len(df.index.symmetric_difference(session_uid.index)) == 0, f"Dataframe and session must have identical indices. Cowardly aborting."




if __name__ == '__main__':
    # path_data_basepath = r'opora/data/'
    path_data_basepath = r'data/'
    # sensors_paths_dict = OrderedDict({
    #     'A_DeviceMotion_data': 'motion',
    #     'B_Accelerometer_data': 'acc',
    #     'C_Gyroscope_data': 'gyro'
    # })
    # activity_type_translation = {
    #     'dws': 'downstairs',
    #     'ups': 'upstairs',
    #     'sit': 'sitting',
    #     'std': 'standing',
    #     'wlk': 'walking',
    #     'jog': 'jogging'
    # }

    # reading and processing all subjects activities from all three sensors to a single df
    all_subject_sensor_data_df, features_cols, label_col = read_all_subjects_activity_data_to_df(path_data_basepath)



