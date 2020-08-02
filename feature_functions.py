import itertools
from collections import OrderedDict
from typing import Union, List
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier


def cal_mag(x: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray], z: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """
    FEATURE EXPLANATION
    According to https://www.kaggle.com/malekzadeh/motionsense-dataset the iPhone-6 was kept in 'the subject's front pocket'.
    However, this implies nothing about the orientation.
    More specifically, it doesn't follow IF ALL SUBJECTS used THE SAME orientation.
    This is why relying an any specific axis might cause subject-specific overfitting.
    To avoid subject-specific overfitting we'll use MAGNITUDE values sqrt(x^2 + y^2 + z^2) for both gyro and accelerometer.
    This will reduce such bias, and may also be easier to analyzed

    Additional information about iPhone sensors axis orientation:
        https://developer.apple.com/documentation/coremotion/getting_raw_accelerometer_events
        https://developer.apple.com/documentation/coremotion/getting_raw_gyroscope_events

    :return: sensor magnitude: sqrt(x^2 + y^2 + z^2)
    """
    return np.sqrt(np.power(np.array(x), 2) + np.power(np.array(y), 2) + np.power(np.array(z), 2))


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

    '''
    Dividing each subject's session to "sub-sessions" called epochs. Each epoch is seconds_per_epoch in length
    Since the dataset might not be divided to exactly in seconds_per_epoch, the leftover is appended to the last epoch
    which can be slightly larger than the result due to that
    '''
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

    all_raw_data_cols = raw_data_cols + ['acc_mag', 'gyro_mag']
    sue_list = list(df[sue_col].unique())
    features_dict_list = []
    for i, sue in enumerate(sue_list):
        print(f"\rCreating features for: {sue} ({i}/{len(sue_list)})                            ", end="")
        cur_series = df.loc[df[sue_col] == sue, all_raw_data_cols]
        stds = cur_series.std()
        avgs = cur_series.mean()

        features_dict = OrderedDict({**stds.add_prefix('std_').to_dict(), **avgs.add_prefix('avg_').to_dict()})
        features_dict['sue'] = sue
        features_dict[label_col] = df.loc[df[sue_col] == sue, label_col].values[0]
        features_dict_list.append(features_dict)
    print("\nDone")

    features_df_final = pd.DataFrame(features_dict_list)
    return features_df_final


def feature_importance_estimate(features: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    """
    :param features: features dataframe
    :param y_true: target labels
    :return: a dataframe (Features, Importance) of the feature importance estimate, using the ExtraTreesClassifier
    """
    model = ExtraTreesClassifier(n_estimators=60, max_depth=5, n_jobs=-1, random_state=90210, verbose=1)
    model.fit(features.values, y_true.values.ravel())
    feature_importance_df = pd.DataFrame({'Feature': list(features), 'Importance': model.feature_importances_})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    return feature_importance_df


