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

# hyper parameter: num_instances, num_samples, test_size
num_samples = 10
test_size = 0.33
num_instances = len(X)
seed = 7

# executing k fold leave one out variation cross validation on data
kfold = model_selection.ShuffleSplit(n_splits=10, test_size=test_size,
									 random_state=seed)

# loading model
model = LogisticRegression()

# checking score
results = model_selection.cross_val_score(model, X, Y, cv=kfold)

# displaying
print("Accuracy: {:.3f}% ({:.3f}%)".format(results.mean()*100, results.std()*100))