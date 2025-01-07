from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
class ModelEvaluation:
  def __init__(self,y_test,y_pred,class_labels):
    self.y_pred = y_pred
    self.y_test = y_test
    self.class_labels = class_labels
    self.report = None

  def evaluate(self):
    self.report = classification_report(self.y_test, self.y_pred, target_names=self.class_labels, output_dict=True)
    print(classification_report(self.y_test, self.y_pred))
    cm = confusion_matrix(self.y_test,self.y_pred)
    cm_df = pd.DataFrame(cm, index=self.class_labels, columns=self.class_labels)
    sns.heatmap(cm_df, cmap = sns.color_palette("coolwarm", as_cmap=True), annot = True,linewidth=.5)
    plt.title("Confusion Matrix")
    plt.show()

  def store(self,model_name,evaluations):
    evaluations = pd.DataFrame(index=self.class_labels)
    precision = [round(self.report[label]['precision']*100,2) for label in self.class_labels]
    recall = [round(self.report[label]['recall']*100,2) for label in self.class_labels]
    f1_score = [round(self.report[label]['f1-score']*100,2) for label in self.class_labels]
    evaluations[model_name+'(Precision)'] = precision
    evaluations[model_name+'(Recall)'] = recall
    evaluations[model_name+'(F1Score)'] = f1_score
    return evaluations
