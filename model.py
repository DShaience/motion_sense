import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from typing import Union, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)



def get_model_and_tuned_params(model_name: str):
    if model_name.lower() == 'LogisticRegression'.lower():
        tuned_parameters = [{'C': [0.01, 0.1, 1, 10]}]
        model = LogisticRegression(random_state=90210, solver='liblinear', multi_class='auto', penalty='l1')
    elif model_name.lower() == 'RandomForest'.lower():
        tuned_parameters = [{'n_estimators': [10, 15, 30],
                             'criterion': ['gini'],
                             'max_depth': [3, 5, 6],
                             'min_samples_split': [10, 15],
                             'min_samples_leaf': [10, 15]
                             # 'class_weight': [{0: 1, 1: 2, 2: 3}]
                             }]
        model = RandomForestClassifier(random_state=90210)
    elif model_name.lower() == 'AdaBoost'.lower():
        model = AdaBoostClassifier(DecisionTreeClassifier())
        tuned_parameters = [{"base_estimator__criterion": ["gini"],
                             "base_estimator__splitter": ["best"],
                             'base_estimator__max_depth': [2, 3, 4],
                             'base_estimator__min_samples_leaf': [10],
                             "n_estimators": [10, 15, 30],
                             "random_state": [90210],
                             'learning_rate': [0.001, 0.01, 0.1]
                             }]
    else:
        raise ValueError("Unsupported classifier type. Cowardly aborting")
    return model, tuned_parameters


def feature_importance_estimate(features: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    model = ExtraTreesClassifier(n_estimators=60, max_depth=5, n_jobs=-1, random_state=90210, verbose=1)
    model.fit(features.values, y_true.values.ravel())
    feature_importance_df = pd.DataFrame({'Feature': list(features), 'Importance': model.feature_importances_})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    return feature_importance_df


def plot_matrix(mat: Union[pd.DataFrame, np.ndarray], fontsz: int, cbar_ticks: List[float] = None, to_show: bool = True):
    """
    :param mat: matrix to plot. If using dataframe, the columns are automatically used as labels. Othereise, matrix is anonymous
    :param fontsz: font size
    :param cbar_ticks: the spacing between cbar ticks. If None, this is set automatically.
    :param to_show: True - plot the figure. Otherwise, close it.
    :return:
    """
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=[8, 8])
    if cbar_ticks is not None:
        ax = sns.heatmap(mat, cmap=cmap, vmin=min(cbar_ticks), vmax=max(cbar_ticks), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticks)
    else:
        ax = sns.heatmap(mat, cmap=cmap, vmin=np.min(np.array(mat).ravel()), vmax=np.max(np.array(mat).ravel()), square=True, linewidths=.5, cbar_kws={"shrink": .5})
        cbar = ax.collections[0].colorbar

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fontsz)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=fontsz)
    if to_show:
        plt.show()
    else:
        plt.close()


def correlation_matrix(df: pd.DataFrame, font_size: int = 10, corrThr: float = None, to_show: bool = True):
    """
    :param df: input dataframe. Correlation matrix calculated for all columns
    :param font_size: font size
    :param toShow: True - plots the figure
    :param corrThr: for easy highlight of significant correlations. Above corrThr, consider the threshold = 1.0. This will highlight the correlative pair
    :param to_show: True - plot the figure. Otherwise, close it.
    :return:
    """
    # Correlation between numeric variables
    cols_numeric = list(df)
    data_numeric = df[cols_numeric].copy(deep=True)
    corr_mat = data_numeric.corr(method='pearson')
    if corrThr is not None:
        assert corr_mat > 0.0, "corrThr must be a float between [0, 1]"
        corr_mat[corr_mat >= corrThr] = 1.0
        corr_mat[corr_mat <= -corrThr] = -1.0

    # print(corr_mat.to_string())

    cbar_ticks = [round(num, 1) for num in np.linspace(-1, 1, 11, dtype=np.float)]  # rounding corrects for floating point imprecision
    plot_matrix(corr_mat, fontsz=font_size, cbar_ticks=cbar_ticks, to_show=to_show)


if __name__ == '__main__':
    rs_subjects_sampling = np.random.RandomState(90210)
    path_data_basepath = r'data/'
    path_additional_data = r'data/data_subjects_info.csv'
    path_features_df = path_data_basepath + 'features_df.p'
    features_df_partial = pickle.load(open(path_features_df, "rb"))

    # rebuilding some meta-data (subjects, epochs)
    label_col = 'label'
    sue_col = 'sue'
    sue_values_list = features_df_partial['sue'].to_list()
    sue_split = [sue.split('_') for sue in sue_values_list]
    subject_list = [f"{x[0]}_{x[1]}" for x in sue_split]
    epoch_list = [f"{x[4]}_{x[5]}" for x in sue_split]
    features_df_partial['sub'] = subject_list
    features_df_partial['epoch'] = epoch_list

    # Adding subjects personal data
    additional_data_df = pd.read_csv(path_additional_data)
    additional_data_df.rename(columns={'code': 'sub'}, inplace=True)
    additional_data_df['sub'] = 'sub_' + additional_data_df['sub'].astype(str).values
    features_df = pd.merge(features_df_partial, additional_data_df, on='sub', how='left')

    # saving feature columns. These columns are the ones that will actually be used by the classifier
    features_cols = [col for col in list(features_df) if col not in [label_col, sue_col, 'sub', 'epoch']]
    feature_importance_df = feature_importance_estimate(features_df[features_cols], features_df[label_col])
    top_important_features = feature_importance_df['Feature'].values[0:21]
    print(feature_importance_df.to_string())
    '''
    ABOUT FEATURES-SET
    While we'll still use all features, it is interesting to not that the additional-data df (age, weight, gender, height)
    were among the *least* informative features.
    This is surprising especially for height and weight which dictate a lot about the physiology of the movement.
    It is also worth noting that these 'additional data' features are very much correlated. For example:
    * high height values will probably correlate with weight (the taller we are, the heavier we are)
    * gender is expected to correlate with both height and weight (females are, on average, smaller and shorter in most human populations).
      [this of course doesn't say anything about specific individuals]
    * age is expected to negatively correlate with height, as we grow shorter with age due to the wear and tear of cartilage in our skeleton
    * age is also expected to correlate with weight, as human metabolism slows does with age, leading to a buildup of excess fat tissues, etc.
    
    For all other features we expect to find high correlations (i.e., most will be uninformative). This is due to several reasons:
    * Many features are derived from the rest, leading to them being dependant. For example, acc-magnitude is dependant on acc_x, acc_y, and acc_z
      Another example: pitch, roll, and yaw are (or can be) derived from accelerometer data
    * The sensor data itself is dependent: a change in one implies a corresponding change in one (or more) of the others.
    
    SUMMARY: the features in this features-set are of very limited value, and can definitely improve with some more
    feature engineering work to create some non-trivial, and less dependant features. 
    
    '''

    unique_subject_list = list(set(subject_list))
    hold_out_subjects = list(rs_subjects_sampling.choice(np.array(unique_subject_list).ravel(), 5))
    train_subjects = [sub for sub in unique_subject_list if sub not in hold_out_subjects]

    '''
    ABOUT TRAIN, TEST and HOLD-OUT
    
    '''

    df_train = features_df[features_df['sub'].isin(train_subjects)]
    df_test = features_df[features_df['sub'].isin(hold_out_subjects)]

