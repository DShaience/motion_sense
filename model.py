import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesRegressor, ExtraTreesClassifier


def feature_importance_estimate(features: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    model = ExtraTreesClassifier(n_estimators=20, max_depth=5, n_jobs=-1, random_state=90210)
    model.fit(features.values, y_true.values.ravel())
    feature_importance_df = pd.DataFrame({'Feature': list(features), 'Importance': model.feature_importances_})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    return feature_importance_df


if __name__ == '__main__':
    LOAD_FROM_PICKLE = True
    path_data_basepath = r'data/'
    path_features_df = path_data_basepath + 'features_df.p'
    features_df = pickle.load(open(path_features_df, "rb"))

    label_col = 'label'
    sue_col = 'session_uid_epoch'
    features_cols = [col for col in list(features_df) if col not in [label_col, sue_col]]
    feature_importance_df = feature_importance_estimate(features_df[features_cols], features_df[label_col])
    print(feature_importance_df.to_string())
    # df['session_uid_epoch'] = df['session_uid'] + '_epoch_' + df['epoch'].astype(str)

