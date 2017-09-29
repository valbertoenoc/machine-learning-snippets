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

# hyper parameter: test_size
test_size = 0.10
seed = 7

# separating train from test and spliting test size using scikit learn
X_train, X_test, Y_train, Y_test = \
model_selection.train_test_split(X,Y,test_size=test_size,random_state=seed)

# loading model
model = LogisticRegression()

# executing model on training data
model.fit(X_train, Y_train)

# checking score
result = model.score(X_test, Y_test)

# displaying
print("Accuracy: {0:.3f}%".format(result*100)) 