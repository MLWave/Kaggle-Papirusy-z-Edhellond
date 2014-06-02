import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import ensemble

if __name__ == '__main__':
  loc_train = "kaggle_hermes\\train.csv"
  loc_test = "kaggle_hermes\\evaluate.csv"
  loc_submission = "kaggle_hermes\\kaggle.hermes.submission.csv"
  #start counter and set random seed
  start = datetime.now()
  np.random.seed(10)
  #fetch dataframes
  df_train = pd.read_csv(loc_train)
  df_test = pd.read_csv(loc_test)
  #shuffle train df to prevent malordered samples
  df_train = df_train.reindex(np.random.permutation(df_train.index))
  #get the feature columns
  feature_cols = [col for col in df_train.columns if col not in ['class']]
  #create a train and test set
  X_train = df_train[feature_cols]
  X_test = df_test[feature_cols]
  #fetch the labels into 'y'
  y = df_train['class']
  #classifier config and fitting
  clf_base = ensemble.RandomForestClassifier(n_estimators=1050,criterion="entropy",max_features=None,random_state=777,n_jobs=-1)
  clf = ensemble.AdaBoostClassifier(clf_base, n_estimators=4, random_state=93)
  print "\nFitting:\n", clf, "\non train set shaped:\n", X_train.shape
  clf.fit(X_train,y)
  #predicting and storing results  
  with open(loc_submission, "wb") as outfile:
    print "\nPredicting on test set shaped:\n", X_test.shape, "\nWriting to:", outfile
    outfile.write("Id,Class\n")
    for e, val in enumerate(clf.predict_proba(X_test)):
      outfile.write( "%s,%f\n" % (float(e+1),float(val[1])))
  print "\nScript running time:", datetime.now()-start