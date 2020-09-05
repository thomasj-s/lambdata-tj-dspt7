from sklearn.pipeline import make_pipeline
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd


def basic_classifier(df, columns, target):

  df = pd.DataFrame(df)  
  df.columns = columns
  df = df.fillna('missing')
  
  cleaner = make_pipeline(
      ce.OrdinalEncoder(),
      SimpleImputer(strategy='mean')
  )
  
  df = cleaner.fit_transform(df)
  df1 = pd.DataFrame(df)
  df1.columns = columns

  train, test = train_test_split(df, train_size=.6)
  val, test = train_test_split(test, train_size=.6)
 
  train = pd.DataFrame(train)
  val = pd.DataFrame(val)
  test = pd.DataFrame(test)
  
  train.columns = columns
  val.columns = columns
  test.columns = columns
  
  xtrain = train.drop(columns=target)
  ytrain = train[target]

  xval = val.drop(columns=target)
  yval = val[target]

  model = RandomForestClassifier(max_depth=5)

  model.fit(xtrain, ytrain)
  guesses = model.predict(xval)

  accuracy = accuracy_score(yval, guesses)
  precision = precision_score(yval, guesses)
  recall = recall_score(yval, guesses)
  f1 = f1_score(yval, guesses)

  return f'accuracy = {accuracy}', f'precision = {precision}', f'recall = {recall}', f'f1 = {f1}'


def times_100(number):
  return number*100
print(times_100(5))


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