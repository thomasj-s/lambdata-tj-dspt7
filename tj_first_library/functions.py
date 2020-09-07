from sklearn.pipeline import make_pipeline
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd



def basic_classifier(df, columns, target):

  '''
     This function is a elementery RandomForest classifier that cleans, splits,
     and predicts on validation set.  Returns metrics accuracy, precision (lab_class = 1),
     recall, and f1.
     
     Takes three arguments : raw dataframe, list of column headers for that dataframe (df.columns), 
     and the binary target column name as a string.

     Fills NaNs with string value 'missing', applies OrdinalEncoder, and applies
     SimpleImputer with strategy 'mean'.
  '''

  # This section defines input as 'dataframe' and fills nan values with
  # string value, 'missing'
  df = pd.DataFrame(df)  
  df = df.fillna('missing')

  # This section defines our basic pipeline
  cleaner = make_pipeline(   
      ce.OrdinalEncoder(),
      SimpleImputer(strategy='mean')
  )
  
  # This section applies pipeline to our dataframe
  df_clean = cleaner.fit_transform(df) 
  df_clean = pd.DataFrame(df_clean)
  df_clean.columns = columns

  # This section aplits our transformed dataframe
  train, test = train_test_split(df_clean, train_size=.6) 
  val, test = train_test_split(test, train_size=.6)
 
  # This section creates a class we can use to derive feature and target
  # sets from our split data sets.  By using this class, we do not
  # have to create a separate x and y group for our train, and val sets.
  class BaseSet:
    def __init__(self, features, target):
      self.df = df
      self.features = features
      self.target = target


  train_base = BaseSet(train.drop(columns=target), train[target])
  val_base = BaseSet(val.drop(columns=target), val[target])
  test_base = BaseSet(test.drop(columns=target), test[target])

  # The remaining code fits an elementary random forest model to our
  # clean data, using the BaseSet class items to predict and report
  # evaluation metrics.
  model = RandomForestClassifier(max_depth = 5)

  model.fit(train_base.features, train_base.target)
  guesses = model.predict(val_base.features)

  accuracy = accuracy_score(val_base.target, guesses) # Calculate metrics
  precision = precision_score(val_base.target, guesses)
  recall = recall_score(val_base.target, guesses)
  f1 = f1_score(val_base.target, guesses)

  return f'accuracy = {accuracy}', f'precision = {precision}', f'recall = {recall}', f'f1 = {f1}'


def times_100(number):
    return number*100
  

def train_validation_test_split(self, features, target,
                                    train_size=0.7, val_size=0.1,
                                    test_size=0.2, random_state=None,
                                    shuffle=True):
        '''
        This function is a utility wrapper around the Scikit-Learn train_test_split that splits arrays or 
        matrices into train, validation, and test subsets.
        Args:
            X (Numpy array or DataFrame): This is a dataframe with features.
            y (Numpy array or DataFrame): This is a pandas Series with target.
            train_size (float or int): Proportion of the dataset to include in the train split (0 to 1).
            val_size (float or int): Proportion of the dataset to include in the validation split (0 to 1).
            test_size (float or int): Proportion of the dataset to include in the test split (0 to 1).
            random_state (int): Controls the shuffling applied to the data before applying the split for reproducibility.
            shuffle (bool): Whether or not to shuffle the data before splitting
        Returns:
            Train, test, and validation dataframes for features (X) and target (y). 
        '''
        X_train_val, X_test, y_train_val, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (train_size + val_size),random_state=random_state, shuffle=shuffle)
        return X_train, X_val, X_test, y_train, y_val, y_test

