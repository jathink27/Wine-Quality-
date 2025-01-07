from sklearn.model_selection import train_test_split
import pandas as pd
from imblearn.over_sampling import SMOTE
class TrainTestSplit:
  def __init__(self, numerical_cols, target_col):
    self.numerical_cols = numerical_cols
    self.target_col = target_col
    self.X_train = pd.DataFrame()
    self.X_test = pd.DataFrame()
    self.y_train = pd.DataFrame()
    self.y_test = pd.DataFrame()

  def split(self, wine_data):
    X_, y_ = SMOTE(random_state = 42).fit_resample(wine_data[self.numerical_cols],
                                               wine_data[self.target_col])
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_,
                                                                            y_,
                                                                            stratify = y_,
                                                                            random_state = 42)

  def data_train(self):
    train = pd.DataFrame()
    train[self.numerical_cols] = self.X_train
    train[self.target_col] = self.y_train
    return train.reset_index()

  def data_test(self):
    test = pd.DataFrame()
    test[self.numerical_cols] = self.X_test
    test[self.target_col] = self.y_test
    return test.reset_index()
