import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from typing import List
import numpy as np
import seaborn as sns
from feature_functions import feature_importance_estimate
from model_and_reports import cm_and_classification_report
sns.set(color_codes=True)


def get_model_and_tuned_params(model_name: str):
    """
    :param model_name: model name as string
    :return: this function is a short basic wrapper to enable this script to support multiple
    classifiers with only minor changes. The function returns a model object and it's tune_parameters dictionary.
    """
    if model_name.lower() == 'LogisticRegression'.lower():
        tuned_parameters = [{'C': [0.01, 0.1, 1, 10]}]
        model = LogisticRegression(random_state=90210, solver='liblinear', multi_class='auto', penalty='l1')
    elif model_name.lower() == 'RandomForest'.lower():
        tuned_parameters = [{'n_estimators': [10, 15, 30],
                             'criterion': ['gini'],
                             'max_depth': [3, 4, 5, 6],
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
                             "n_estimators": [10, 15, 20, 30],
                             "random_state": [90210],
                             'learning_rate': [0.001, 0.01, 0.1]
                             }]
    else:
        raise ValueError("Unsupported classifier type. Cowardly aborting")
    return model, tuned_parameters


if __name__ == '__main__':
    rs_subjects_sampling = np.random.RandomState(90210)  # random-state for subject sampling
    path_data_basepath = r'data/'
    path_additional_data = r'data/data_subjects_info.csv'
    path_features_df = path_data_basepath + 'features_df.p'
    features_df_partial = pickle.load(open(path_features_df, "rb"))

    # re-creating some meta-data (subjects, epochs)
    label_col = 'label'
    activity_labels_list = ['ups', 'dws', 'wlk', 'std', 'sit', 'jog']
    sue_col = 'sue'
    sue_values_list = features_df_partial['sue'].to_list()
    sue_split = [sue.split('_') for sue in sue_values_list]
    subject_list = [f"{x[0]}_{x[1]}" for x in sue_split]  # subject designation (sub_x, sub_y, etc.)
    epoch_list = [f"{x[4]}_{x[5]}" for x in sue_split]  # epoch number (epoch_1, epoch_2, ..)
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
    top_important_features = feature_importance_df['Feature'].values[0:20]  # Top 20 most important features
    print(feature_importance_df.to_string())
    features_to_user_cols = top_important_features
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
    We'll still use the full feature-set for now, since the model is very light to calculate.
    Usually, we'll then try to reduce features-set by using feature_importance (see top_important_features)
    or other feature selection methods. The easier in this script would be to use the 10-20 "top" most "important".
    It would also be possible to forward/backward select features, or use other standard methodologies.
    
    '''

    unique_subject_list = list(set(subject_list))
    hold_out_subjects = list(rs_subjects_sampling.choice(np.array(unique_subject_list).ravel(), 5))
    train_subjects = [sub for sub in unique_subject_list if sub not in hold_out_subjects]

    '''
    ABOUT TRAIN, TEST and HOLD-OUT
    The dataset is quite small: there aren't many repetitions of each action activity. Between 2-3 per activity-type.
    In addition to that, there's a small number of users (24), which means that each activity has 48-72 samples, which is
    not a whole lot, when thinking this should be enough for train, test AND cross-validation of the model for SIX activities.
    So, we expect that such dataset is going to be very prone to overfitting, under such conditions.
    
    To address this issues, I selected the following strategies:
    (1) Divide each user to multiple ~10-seconds epochs. This increases the number of samples that can be used.
    While they belong to the same session, each epoch is a slightly varied part of the session,  
    in which the activity occurs. Simply stated, a running session of 60 seconds is broken down to
    6 epochs of 10-seconds runs, each one labeled as 'running'.
    This will give us more leeway when doing cross-validation
    
    (2) Hold-out data
    It is safe to assume (as well as from my personal experience) that humans are 'creatures of habit': that is,
    the expected variance between sessions of the same subjects will be lower than the variance with other subject, for the
    same activity. 
    Due to this *assumption* I decided that the best was to test the model is over ~20% (5 users) which are left as hold-out data.
    The rest of the 19 subjects are used for train/cross-validation
    
    (3) Cross-validation: Boot-strap cross validation (aka, Monte-Carlo cross validation)
    Since the number of users remaining in the train-set (19) and activities (6) is small, it becomes a problem to select
    and train-set that's sufficiently representative of the problem, and isn't too biased for one subject or another.
    This problem is compounded by the fact the some subjects recorded lengthier sessions than others.
    To overcome this problem I'll use a boot-strap approach, in which I'll create N independent pseudo-datasets, 
    sampling ~65% of the data (without replacement, though good practice is usually WITH replacement). The rest will be used as test-set.
    I'll then train N classifiers and average their performance. 
    A good classifier is expected to have (GOOD performance) AND (LOW performance variability)
    
    One potential problem remains, is that even so, some pseudo-datasets might not 
    contain ALL activities, due to randomness. To mitigate this risk, we'll use a significant amount of the data
    for train/test (65%/35%)  
    '''

    df_train = features_df[features_df['sub'].isin(train_subjects)].copy(deep=True)
    df_train.reset_index(inplace=True)
    df_holdout = features_df[features_df['sub'].isin(hold_out_subjects)].copy(deep=True)
    df_holdout.reset_index(inplace=True)

    # Scaling and creating input for classifier
    scaler = StandardScaler()
    #   Train
    X_train = df_train[features_to_user_cols]
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    y_train = df_train[label_col].values
    #   Hold-out
    X_holdout_scaled = scaler.transform(df_holdout[features_to_user_cols])
    y_holdout = df_holdout[label_col].values

    # Monte-carlo cross-validation
    print("Creating CV (using 10 bootstrap datasets)")
    mccsv = ShuffleSplit(n_splits=10, test_size=0.35, random_state=90210)
    cv: List[tuple] = []
    for train_index, test_index in mccsv.split(df_train):
        cv.append((train_index, test_index))
    print("Done")

    print("Start training model")
    # Model and grid-search
    # Example for using other classifiers supported by this script
    #     model, tuned_parameters = get_model_and_tuned_params(model_name='LogisticRegression')
    #     model, tuned_parameters = get_model_and_tuned_params(model_name='RandomForest')
    model, tuned_parameters = get_model_and_tuned_params(model_name='AdaBoost')
    gs = GridSearchCV(model, tuned_parameters, scoring='f1_macro', n_jobs=-1, cv=cv, refit=True, verbose=1)
    gs.fit(X_train_scaled, y_train)

    '''
    Normally, I'd defined some threshold to accept a model only for:
        (1) STD(performance) < thr_1 usually, 0.05, or 0.025 depending on the dataset
        (2) performance >= thr_2 just to filter-out low performing models
    In this cases, since STD was very low, and the model was well-behaved (when comparing train and hold-out), I felt that
    the gs best estimator is good enough to leave it at that.
    
    Note about the results:
    While both train and hold-out are characterized in high-recall/high-precision, it is both
    interesting and encouraging to note that the *MIS-CLASSIFICATIONS MAKE SENSE*.
    The model may confuse 'upstairs' and 'downstairs'. These are indeed very similar actions.
    Also, 'walking' can be confused with either of the other dynamic activities (jogging, upstairs, downstairs), 
    which again, makes sense.
    
    Though I can offer no support for this at this time, I *suspect* that some of the confusion MAY be,
    in part, due to inconsistent orientation of the phone worn by some subjects. 
    In such a case, upstairs and downstairs may indeed look similar, if the phone was, for example, transposed.
    Some of the features can be affected by this (especially separate-raw-axes data from accelerometer and gyroscope).   
    '''

    # Best GS model
    best_idx = gs.best_index_
    clf = gs.best_estimator_
    print(f"\nBest estimator params:\n\tParams: {gs.best_params_}\n\tBest Score: {gs.best_score_}\n")
    print(f"GridSearch mean-test-score: {gs.cv_results_['mean_test_score'][best_idx]}")
    print(f"GridSearch std-test-score: {gs.cv_results_['std_test_score'][best_idx]}")

    print("\n========================================================\n Train Classification Report\n========================================================")
    y_train_pred = clf.predict(X_train_scaled)
    cm_and_classification_report(y_train, y_train_pred, labels=activity_labels_list)

    # Holdout data
    print("\n========================================================\n Hold-out Classification Report\n========================================================")
    y_holdout_pred = clf.predict(X_holdout_scaled)
    cm_and_classification_report(y_holdout, y_holdout_pred, labels=activity_labels_list)
    print("")

    '''
    -------------------------------------------------------------------------------------------
    Appendix
    -------------------------------------------------------------------------------------------
    (A) Feature Importance
    -------------------------------------------------------------------------------------------
                   Feature  Importance
        avg_attitude.pitch    0.062134
             avg_gravity.y    0.052514
                avg_gyro_y    0.050069
        std_rotationRate.x    0.048946
                std_gyro_x    0.047880
              avg_gyro_mag    0.046901
               std_acc_mag    0.046551
                std_gyro_y    0.046098
                 avg_acc_y    0.042942
        std_rotationRate.y    0.039323
             std_gravity.z    0.037886
        avg_rotationRate.y    0.037693
               avg_acc_mag    0.036843
    std_userAcceleration.x    0.036676
              std_gyro_mag    0.035617
             avg_gravity.z    0.033940
                 std_acc_y    0.030779
          std_attitude.yaw    0.029296
                 avg_acc_z    0.027541
    std_userAcceleration.z    0.026505
    std_userAcceleration.y    0.024867
                 std_acc_x    0.021008
             std_gravity.x    0.020824
        std_rotationRate.z    0.020576
        std_attitude.pitch    0.012475
                 std_acc_z    0.012253
             std_gravity.y    0.009526
                std_gyro_z    0.008841
    avg_userAcceleration.z    0.007490
                 avg_acc_x    0.006585
             avg_gravity.x    0.006415
        avg_rotationRate.z    0.005942
          avg_attitude.yaw    0.005618
         avg_attitude.roll    0.003584
    avg_userAcceleration.y    0.003378
         std_attitude.roll    0.003195
                avg_gyro_z    0.003151
    avg_userAcceleration.x    0.003134
                    gender    0.001986
                    height    0.001434
        avg_rotationRate.x    0.000527
                    weight    0.000405
                       age    0.000348
                avg_gyro_x    0.000306

    While I didn't end-up using it, it would be easy to include only top "most important" features, using this table
    -------------------------------------------------------------------------------------------

    -------------------------------------------------------------------------------------------
    (B) Results (model predictions)
    -------------------------------------------------------------------------------------------
    This part includes some train/hold-out classification report and results.
    Best estimator params:
        Params: {'base_estimator__criterion': 'gini', 'base_estimator__max_depth': 4, 'base_estimator__min_samples_leaf': 10, 'base_estimator__splitter': 'best', 
        'learning_rate': 0.1, 'n_estimators': 30, 'random_state': 90210}
        Best Score: 0.9635605049173435
    
    GridSearch mean-test-score: 0.9635605049173435
    GridSearch std-test-score: 0.008212684235527516
    
    ========================================================
     Train Classification Report
    ========================================================
         t/p    ups   dws   wlk   std   sit   jog 
          ups   213     4     0     0     0     0 
          dws     0   176     0     0     0     0 
          wlk     0     0   512     0     0     0 
          std     0     0     0   478     0     0 
          sit     0     0     0     0   530     0 
          jog     0     0     0     0     0   196 
    
                  precision    recall  f1-score   support
    
             dws      0.978     1.000     0.989       176
             jog      1.000     1.000     1.000       196
             sit      1.000     1.000     1.000       530
             std      1.000     1.000     1.000       478
             ups      1.000     0.982     0.991       217
             wlk      1.000     1.000     1.000       512
    
        accuracy                          0.998      2109
       macro avg      0.996     0.997     0.997      2109
    weighted avg      0.998     0.998     0.998      2109
    
    
    
    ========================================================
     Hold-out Classification Report
    ========================================================
         t/p    ups   dws   wlk   std   sit   jog 
          ups    56     0     3     0     0     0 
          dws     0    49     1     0     0     0 
          wlk     0     0   137     0     0     0 
          std     0     0     0   114     0     0 
          sit     0     0     0     0   123     0 
          jog     0     0     0     0     0    45 
    
                  precision    recall  f1-score   support
    
             dws      1.000     0.980     0.990        50
             jog      1.000     1.000     1.000        45
             sit      1.000     1.000     1.000       123
             std      1.000     1.000     1.000       114
             ups      1.000     0.949     0.974        59
             wlk      0.972     1.000     0.986       137
    
        accuracy                          0.992       528
       macro avg      0.995     0.988     0.992       528
    weighted avg      0.993     0.992     0.992       528

    '''
