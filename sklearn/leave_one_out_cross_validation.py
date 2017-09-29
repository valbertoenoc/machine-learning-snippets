import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

# loading data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
filename = '../data/pima-indians-diabetes.data'
names =['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(filename, names=names)

# loading to arrays
data = dataframe.values
print(data.shape)
X = data[:, 0:8] 
Y = data[:, 8]

# hyper parameter: num_instances
num_instances = len(X)
seed = 7

# executing k fold leave one out variation cross validation on data
loocv = model_selection.LeaveOneOut()

# loading model
model = LogisticRegression()

# checking score
results = model_selection.cross_val_score(model, X, Y, cv=loocv)

# displaying
print("Accuracy: {:.3f}% ({:.3f}%)".format(results.mean()*100, results.std()*100))