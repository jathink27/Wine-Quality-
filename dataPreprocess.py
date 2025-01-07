from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
class DataPreprocess:
  def __init__(self, train, test, numerical_cols, target_col):
    self.train = train
    self.test = test
    self.numerical_cols = numerical_cols
    self.target_col = target_col

  def handlingMissingValues(self):
    if self.train[self.numerical_cols].isna().sum().sum() == 0:
      return "Since there are no NULL values, we will proceed with other processes....."
    null_col = self.train.columns[self.train.isna().any()].tolist()
    for col in null_col:
      self.train[col] = self.train[col].fillna(self.train[col].median()[0])
    return self.train

  def transformData(self):
    le = LabelEncoder()
    le.fit(self.train[self.target_col])
    self.train[self.target_col] = le.transform(self.train[self.target_col])
    self.test[self.target_col] = le.transform(self.test[self.target_col])
    return self.train, self.test, le.classes_

  def dataScaling(self):
    ss = StandardScaler()
    ss.fit(self.train[self.numerical_cols])
    self.train[self.numerical_cols] = ss.transform(self.train[self.numerical_cols])
    self.test[self.numerical_cols] = ss.transform(self.test[self.numerical_cols])
    return self.train

  def eda(self):
    acid = ['fixed acidity','volatile acidity','citric acid','pH']
    chemical_compounds = ['residual sugar','chlorides','density','alcohol']
    sulphur = ['free sulfur dioxide','total sulfur dioxide','sulphates']
    print(":::Box-Plot:::")
    sns.boxplot(self.train[acid])
    plt.title('BoxPlot for various acid related properties')
    plt.grid()
    plt.show()
    sns.boxplot(self.train[chemical_compounds])
    plt.title('Box-Plot for various chemical compounds')
    plt.grid()
    plt.show()
    sns.boxplot(self.train[sulphur])
    plt.title('Box-Plot for various sulphur related properties')
    plt.grid()
    plt.show()
    print(":::Co-relation Matrix:::")
    sns.heatmap(self.train[self.numerical_cols].corr())
    plt.title("Co-relation Matrix for all the chemical properties of the wine")
    plt.show()
