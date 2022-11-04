import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as NB
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.colors import ListedColormap
import seaborn as sns

matplotlib_axes_logger.setLevel('ERROR')


df = pd.read_csv("mushrooms.csv")



print(df.shape)
print(df.head)
print(df.isnull().sum())
df = df.drop('veil-type',axis=1)


#Seperating target variable and predictor variables.



X=df.drop('class',axis=1) 
y=df['class'] 

np.random.seed(40)

count = df['class'].value_counts()
sns.barplot(count.index, count.values) 


plt.ylabel('Count')
plt.xlabel('Class')
plt.title('Number of poisonous/edible mushrooms')
plt.show() 


#Enconde the data, change catagorical to numbers
Encoder_X = LabelEncoder() 
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)

corr_matrix = X.corr()
print(corr_matrix)


plt.figure(figsize=(25,15))
plt.title('Correlation Heatmap of Mushroom Dataset')
a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show() 



#Get Dummy Data for X features
X=pd.get_dummies(X,columns=X.columns,drop_first=True)
    
#Standardize the data
sc = StandardScaler()
X=sc.fit_transform(X)




#Apply PCA
print(type(X))
pca = PCA(n_components=2)
components = pca.fit_transform(X)

#Train the data
X_train, X_test, y_train, y_test = train_test_split(components, y, random_state=0)

#Put PCA into a pandas dataframe
pca_df = pd.DataFrame(data = components
             , columns = ['principal component 1', 'principal component 2'])
#Check Tail of PCA
print(pca_df.tail())
#plot PCA
plt.figure()
plt.figure(figsize=(10,15))
plt.yticks(fontsize=25)
plt.xticks(fontsize=25)
plt.xlabel('PC 1',fontsize=20)
plt.ylabel('PC 2',fontsize=20)
plt.title("Principal Component Analysis of Mushrooms",fontsize=20)
targets = ['p', 'e']
colors = ['purple', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = df['class'] == target
    plt.scatter(pca_df.loc[indicesToKeep, 'principal component 1']
               , pca_df.loc[indicesToKeep, 'principal component 2'], c = color, s = 60)

plt.legend(targets,prop={'size': 20})

#create visulisation function   
########
def mushroom_test(model,model_name):
    plt.figure(figsize=(10,15))
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('purple', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('purple', 'green'))(i), label = j)
    plt.title("%s Test Set" %(model_name))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()

#Create evaulation function
def evaulation(model,X_test,y_test,model_name):
    print("#-----------------------------------#")
    print("Test Results for " + model_name)
    print("Accuracy Score: ") 
    print(accuracy_score(y_test,model.predict(X_test)))
    print("Cross Validation Score: ")
    cv_acc=cross_val_score(model, X_train,y_train,cv=5)
    print(cv_acc)
    print("Mean Score: ")
    print(cv_acc.mean())
    print("Standard Deviation: ")
    print(cv_acc.std())
                        
    print("#-----------------------------------#")

   
#Logistic Regression test model
log_model = LogisticRegression()
fitted_log = log_model.fit(X_train,y_train)
fitted_log
mushroom_test(log_model,'Logistic Regression')
evaulation(log_model, X_test, y_test, 'Logistic Regression')

#confusion matrix
matrix = plot_confusion_matrix(fitted_log, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for Logistic Regression')
plt.show(matrix)
plt.show()

#SVC test model
svc_model = SVC(kernel='rbf',random_state=42)
fitted_svm = svc_model.fit(X_train,y_train)
fitted_svm
mushroom_test(svc_model,'SVC')
evaulation(svc_model, X_test, y_test, 'SVC')

#confusion matrix
matrix = plot_confusion_matrix(fitted_svm, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for SVM')
plt.show(matrix)
plt.show()


#DT test model
dt_model = DT(criterion='entropy',random_state=42)
fitted_dt = dt_model.fit(X_train,y_train)
fitted_dt
mushroom_test(dt_model, 'DecisionTree Classifier')
evaulation(dt_model, X_test, y_test, 'DecisionTree Classifier')

#confusion matrix
matrix = plot_confusion_matrix(fitted_dt, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for Decision Tree')
plt.show(matrix)
plt.show()


#KNN test model
knn_model = KNN()
fitted_knn = knn_model.fit(X_train,y_train)
fitted_knn
mushroom_test(knn_model, 'K Nearest Neighbors (K-NN)')
evaulation(knn_model, X_test, y_test, 'K Nearest Neighbors (K-NN)')

#confusion matrix
matrix = plot_confusion_matrix(fitted_knn, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for KNN')
plt.show(matrix)
plt.show()


#NB test model
nb_model = NB()
fitted_nb = nb_model.fit(X_train,y_train)
fitted_nb
mushroom_test(nb_model, 'Naive Bayes')
evaulation(nb_model, X_test, y_test, 'Naive Bayes')

#confusion matrix
matrix = plot_confusion_matrix(fitted_nb, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for Naive Bayes')
plt.show(matrix)
plt.show()


#RF test model
rf_model = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)
fitted_rf = rf_model.fit(X_train, y_train)
fitted_rf
mushroom_test(rf_model, 'Random Forest')
evaulation(rf_model, X_test, y_test, 'Random Forest')

#confusion matrix
matrix = plot_confusion_matrix(fitted_rf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for Random Forest')
plt.show(matrix)
plt.show()
