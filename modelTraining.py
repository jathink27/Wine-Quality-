from sklearn.multiclass import OneVsRestClassifier
class ModelTraining:
  def __init__(self,classifier_model, hyperparameters):
    self.model = None
    self.classifier_model = classifier_model
    self.hyperparameters = hyperparameters

  def train_model(self,X,y):
    self.classifier_model.set_params(**self.hyperparameters)
    print(self.classifier_model.get_params())
    self.model = OneVsRestClassifier(self.classifier_model).fit(X,y)

  def predict(self,y):
    return self.model.predict(y)
