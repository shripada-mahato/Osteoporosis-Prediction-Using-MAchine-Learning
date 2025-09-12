import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("osteoporosisfinal - osteoporosisfinal (1).csv")

# Drop unnecessary columns and NaN values
data_final = data.drop(["Id", "Race", "Medications"], axis=1)
data_final = data_final.dropna()

print(data_final)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Encode categorical values
data_final['Gender'] = le.fit_transform(data_final['Gender'])
data_final['Hormone'] = le.fit_transform(data_final['Hormone'])
data_final['FHistory'] = le.fit_transform(data_final['FHistory'])
data_final['Weight'] = le.fit_transform(data_final['Weight'])
data_final['CalciumIn'] = le.fit_transform(data_final['CalciumIn'])
data_final['Smoking'] = le.fit_transform(data_final['Smoking'])
data_final['MedCondition'] = le.fit_transform(data_final['MedCondition'])
data_final['Fractures'] = le.fit_transform(data_final['Fractures'])
data_final['Activity'] = le.fit_transform(data_final['Activity'])
print(data_final)

# Features and target
x = data_final.iloc[:, :-1].values
y = data_final.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)


# Predict on test set
y_pred = model.predict(x_test)
# Example input prediction
inp = [[12,0,1,1,1,1,0,1,1,1]]
print("Sample Prediction:", model.predict(inp))


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cmknn = confusion_matrix(y_test, y_pred)
plt.title("KNN Confusion Matrix")
plt.subplot(4,2,1)
sns.heatmap(cmknn, annot=True, fmt="d", cmap="Blues")
plt.show()


#LogicticRegression
from sklearn.linear_model import LogisticRegression
logm=LogisticRegression()
logm.fit(x_train,y_train)
logm_pred=logm.predict(x_test)
cmlogm=confusion_matrix(y_test,logm_pred)
plt.title("Logistic Regression Confusion Matrix")
plt.subplot(4,2,2)
sns.heatmap(cmlogm,annot=True)

#finding Accuracy using Naive_Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nb_pred = nb.predict(x_test)
acnb = accuracy_score(y_test,nb_pred)
print("naive bayes"+str(acnb))
plt.subplot(4,2,5)
plt.title("Naive Bayes")
cmnb = confusion_matrix(y_test,nb_pred)
sns.heatmap(cmnb, annot=True)

#finding Accuracy using SVM(Support Vector Machine)
from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(x_train,y_train)
svm_pred = svm.predict(x_test)
svmac = accuracy_score(y_test,svm_pred)
print("SVM accuracy: "+str(svmac))
cmsvm = confusion_matrix(y_test,svm_pred)
plt.subplot(4,2,6)
plt.title("SVM Confusion Matrix")
sns.heatmap(cmsvm, annot=True)
plt.show()