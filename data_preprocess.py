import pickle
import pandas as pd
# import seaborn as sns
# sns.set(color_codes=True)
from typing import List, Tuple
from feature_functions import add_session_epochs, calculated_features_df


def read_sensor_data_by_subject(base_path: str, cur_activity: str, cur_sub: str) -> Tuple[pd.DataFrame, List[str], str]:
    """
    :param base_path: data base path
    :param cur_activity: activity designation and number
    :param cur_sub: subject designation
    :return: a dataframe where all three sensor sources are merged (inner). This (for the sake of simplicity) retains
    observations that exist on all three sensors.
    The dataframe also includes activity and subject as meta-data.
    We also construct a label column using cur_activity
    In addition, this function also returns a list of raw_data_columns, which will later on help differentiate between raw-data, meta-data, and label
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
    sub_acc_df.rename(columns={'x': 'acc_x', 'y': 'acc_y', 'z': 'acc_z'}, inplace=True)

    # read gyro
    sub_gyro_df = pd.read_csv(sensor_data_paths[2])
    del sub_gyro_df[sub_gyro_df.columns[0]]
    sub_gyro_df.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}, inplace=True)

    # Joining all subject's sensor data
    '''
    # Note: Different sensors have (slightly) different number of observations, which is totally normal 
    # for sensors data. For simplicity's sake I used only observations that exists across all three sensors
    # and threw out any excess information one of the sensors have.
    # In sampling rate of 50Hz, this usually resulted in losing less than 1 sec of data
    '''
    cur_sub_sensors_data_df = pd.concat([sub_motion_df, sub_acc_df, sub_gyro_df], axis=1, join='inner')
    raw_data_cols = list(cur_sub_sensors_data_df)

    label_col = 'label'
    cur_sub_sensors_data_df[label_col] = cur_activity.split('_')[0]
    cur_sub_sensors_data_df['subject'] = cur_sub
    cur_sub_sensors_data_df['activity'] = cur_activity
    cur_sub_sensors_data_df['session_uid'] = f"{cur_sub}_{cur_activity}"  # session unique identifier
    cur_sub_sensors_data_df['ts'] = range(0, len(cur_sub_sensors_data_df))
    return cur_sub_sensors_data_df, raw_data_cols, label_col


def read_all_subjects_activity_data_to_df(base_path: str) -> Tuple[pd.DataFrame, List[str], str]:
    """
    :param base_path: dataset base-path. Hard-coded assumption that directory structure is the same as provided in
    the original dataset:
        https://github.com/mmalekzadeh/motion-sense/
    :return: dataframe containing feature, meta-data and label. It also returns columns names lists: raw_data, and label column name
    """
    activity_sets_list = ['dws_1', 'dws_2', 'dws_11', 'jog_9', 'jog_16', 'sit_5', 'sit_13', 'std_6', 'std_14', 'ups_3', 'ups_4', 'ups_12', 'wlk_7', 'wlk_8', 'wlk_15']
    subs_list = ['sub_' + str(i) for i in range(1, 25)]
    all_subject_sensor_data_list = []

    raw_data_cols = label_col = None
    for action in activity_sets_list:
        print(f"Analysing action: {action}")
        for sub in subs_list:
            print(f"\r\tSubject: {sub}", end="")
            # subject_sensor_data_df, raw_data_cols, meta_data_cols_list, label_col = read_sensor_data_by_subject(base_path, action, sub)
            subject_sensor_data_df, raw_data_cols, label_col = read_sensor_data_by_subject(base_path, action, sub)
            all_subject_sensor_data_list.append(subject_sensor_data_df)
        print("")
    print("Done")
    print("Joining all users and sensors data... ")
    all_subject_sensor_data = pd.concat(all_subject_sensor_data_list, sort=False, ignore_index=True)
    all_subject_sensor_data.sort_values(by=['session_uid', 'ts'], inplace=True)
    print("Done")
    return all_subject_sensor_data, raw_data_cols, label_col


if __name__ == '__main__':
    path_data_basepath = r'data/'
    LOAD_FROM_PICKLE = False
    # reading and processing all subjects activities from all three sensors to a single df
    path_subjects_sensor_data = path_data_basepath + 'subjects_sensor_data.p'
    path_subjects_sensor_raw_cols = path_data_basepath + 'subjects_raw_cols.p'
    path_subjects_label_col = path_data_basepath + 'label_col.p'
    path_features_df = path_data_basepath + 'features_df.p'

    print("Loading all subjects and all sensors data")
    if not LOAD_FROM_PICKLE:
        all_subject_sensor_data_df, raw_data_cols, label_col = read_all_subjects_activity_data_to_df(path_data_basepath)
        all_subject_sensor_data_df = add_session_epochs(all_subject_sensor_data_df, 'session_uid')
        pickle.dump(all_subject_sensor_data_df, open(path_subjects_sensor_data, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(raw_data_cols, open(path_subjects_sensor_raw_cols, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(label_col, open(path_subjects_label_col, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        all_subject_sensor_data_df = pickle.load(open(path_subjects_sensor_data, "rb"))
        raw_data_cols = pickle.load(open(path_subjects_sensor_raw_cols, "rb"))
        label_col = pickle.load(open(path_subjects_label_col, "rb"))
    print("Done\n")

    # Generating some (very basic and generic features) over all raw data, in each epoch
    metadata_cols = [col for col in list(all_subject_sensor_data_df) if col not in raw_data_cols and col != label_col]
    print("Calculating features (this may take some time. Go grab a coffee. Read the news. Take a stroll outside!")
    if not LOAD_FROM_PICKLE:
        features_df = calculated_features_df(all_subject_sensor_data_df, raw_data_cols, label_col)
        pickle.dump(features_df, open(path_features_df, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        features_df = pickle.load(open(path_features_df, "rb"))
    print("Done\n")


