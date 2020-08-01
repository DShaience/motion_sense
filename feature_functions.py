from typing import Union
import numpy as np
import pandas as pd


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


