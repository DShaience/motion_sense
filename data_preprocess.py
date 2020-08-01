import pickle
import pandas as pd
import numpy as np
# import seaborn as sns
# sns.set(color_codes=True)
from typing import List, Tuple
import itertools
from collections import OrderedDict
from feature_functions import cal_mag


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


def add_session_epochs(df: pd.DataFrame, session_uid_col: str, sampling_rate_hz: int = 50, seconds_per_epoch: int = 10) -> pd.DataFrame:
    """
    :param df: input dataframe
    :param session_uid_col: session unique identifier to help locate samples belonging to the same
    :param sampling_rate_hz: sensors sampling rate. It is provided as 50Hz in the dataset description
    :param seconds_per_epoch: How many seconds to include per epoch
    :return: This function divides sessions into epochs. We do that since the number of users X activities is rather low.
    By subdividing the session, we may increase the number of observations per each action.
    This will make it easier to divide to train and test.
    NOTE: this function modifies the original df (but this is a quick&dirty exercise, and the original df won't be used downstream in any case)
    """
    df['epoch'] = None
    session_uids_list = list(set(df[session_uid_col].values))

    # Dividing each user's session to "sub-sessions" called epochs. Each epoch is seconds_per_epoch in length
    # Since the dataset might not be divided to exactly in seconds_per_epoch, the leftover is appended to the last epoch
    # which can be slightly larger than the result due to that
    for s_uid in session_uids_list:
        n = np.sum(df[session_uid_col] == s_uid)
        seconds_total = n / sampling_rate_hz
        n_epochs = int(seconds_total // seconds_per_epoch)
        obs_per_epoch = seconds_per_epoch * sampling_rate_hz

        epochs_vec_lists = [[i] * obs_per_epoch for i in range(0, n_epochs)]
        total = len(epochs_vec_lists) * obs_per_epoch
        if total < n:
            epochs_vec_lists.append(([n_epochs - 1] * (n - total)))

        epochs_vec = list(itertools.chain(*epochs_vec_lists))
        df.loc[df[session_uid_col] == s_uid, 'epoch'] = np.array(epochs_vec)

    assert df['epoch'].isna().sum() == 0, "Error. At this point no epoch should be unassigned, yet, some epochs are None"
    # Adding a unique identifier for a session's epoch
    df['session_uid_epoch'] = df['session_uid'] + '_epoch_' + df['epoch'].astype(str)
    return df


def calculated_features_df(df: pd.DataFrame, raw_data_cols: List[str], label_col: str) -> pd.DataFrame:
    """
    adds features to the df.
    Unique session/epoch identifier is hardcoded 'session_uid_epoch' (aka SUE)
    Some features will be added per SUE. While not ideal for some features,
    this will prevent leakage of information between epochs.
    Features generally divide to two classes:
    * transformations: features that are a combination of values per specific column. For example, Acc-magnitude: sqrt(x^2 + y^2 + z^2). All values exist in the same row
    * aggregations: features that may include data for observations in other rows. For example, dx/dt is such as feature, since it requires information from previous row
    Generally, for efficiency's sake, transformation will be applied on the entire dataframe. Aggregations will be either per SUE, or per session_uid, depending on the case

    :return: a new dataframe, that contains ONE row per SUE. This is a row of features, rather than the incoming RAW data
    """
    sue_col = 'session_uid_epoch'

    df['acc_mag'] = cal_mag(df['acc_x'], df['acc_y'], df['acc_z'])
    df['gyro_mag'] = cal_mag(df['gyro_x'], df['gyro_y'], df['gyro_z'])

    # extra_data_df = df[['subject', 'activity', 'session_uid', label_col]].drop_duplicates()
    # subject activity   session_uid    ts epoch     session_uid_epoch label
    all_raw_data_cols = raw_data_cols + ['acc_mag', 'gyro_mag']

    sue_list = list(df[sue_col].unique())
    features_dict_list = []
    for i, sue in enumerate(sue_list):
        print(f"Creating features for: {sue} ({i}/{len(sue_list)})")
        features_dict = OrderedDict()
        for raw_data_col in all_raw_data_cols:
            print(f"\r\tStandard features for: {raw_data_col}                            ", end="")
            cur_series = df.loc[df[sue_col] == sue, raw_data_col]
            features_dict['std_' + raw_data_col] = np.std(cur_series.values)
            features_dict['avg_' + raw_data_col] = np.average(cur_series.values)

        features_dict['sue'] = sue
        features_dict[label_col] = df.loc[df[sue_col] == sue, label_col].values[0]
        features_dict_list.append(features_dict)
        print("\nDone")

    features_df_final = pd.DataFrame(features_dict_list)
    return features_df_final


if __name__ == '__main__':
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

    LOAD_FROM_PICKLE = True
    # reading and processing all subjects activities from all three sensors to a single df
    path_subjects_sensor_data = path_data_basepath + 'subjects_sensor_data.p'
    path_subjects_sensor_raw_cols = path_data_basepath + 'subjects_raw_cols.p'
    path_subjects_label_col = path_data_basepath + 'label_col.p'
    path_features_df = path_data_basepath + 'features_df.p'
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

    # print("")
    metadata_cols = [col for col in list(all_subject_sensor_data_df) if col not in raw_data_cols and col != label_col]
    if not LOAD_FROM_PICKLE:
        features_df = calculated_features_df(all_subject_sensor_data_df, raw_data_cols, label_col)
        pickle.dump(features_df, open(path_features_df, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        features_df = pickle.load(open(path_features_df, "rb"))
