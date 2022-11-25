#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/Eundding/OSS_Assignment.git

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

def load_dataset(dataset_path): #Clear
	#Load the csv file at the given path into the pandas DataFrame and return the DataFrame
	df = pd.read_csv(dataset_path)
	return df

def dataset_stat(dataset_df):
	feature = len(dataset_df.columns) - 1 
	zero_cnt, one_cnt = 0, 0 # class 0, 1을 각각 세는 변수

	for i in range(len(dataset_df)):
		if dataset_df['target'][i] == 0:
			zero_cnt += 1
		else:
			one_cnt += 1
	return feature, zero_cnt, one_cnt

def split_dataset(dataset_df, testset_size): 
	data = dataset_df.drop(['target'], axis=1)
	target = dataset_df['target']
	x_train,  x_test, y_train, y_test = train_test_split(data, target, test_size=testset_size)
	return x_train,  x_test, y_train, y_test


def decision_tree_train_test(x_train, x_test, y_train, y_test): 
	model = DecisionTreeClassifier()
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test) # 예측값
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	model = RandomForestClassifier()
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test) # 예측값
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	model = make_pipeline(
		StandardScaler(),
		SVC()
	)
	model.fit(x_train, y_train)
	y_pred = model.predict(x_test) # 예측값
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)