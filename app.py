import streamlit as st
import numpy as np 
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from settings import DATASET_DIR as dataset
from settings import IMAGE_DIR as img

#for visualization
from io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

def main():
	html_temp="""
	<div style="background-color:yellow;padding:13px">
	<h1 style="color:black;text-align:center;">Decision Tree Algorithm Implementation</h1>
	</div>
	"""

	#displaying the frontend aspect
	st.markdown(html_temp, unsafe_allow_html=True)

	df = pd.read_csv('dataset/drug200.csv')

	st.sidebar.title("Evaluating the Algorithm")
	st.sidebar.subheader("View dataset")
	num = st.sidebar.number_input("Enter number of data", 5, 30)
	st.write(df.head(num))
	st.write("Total size of data: ", df.shape)

	#preprocessing data
	X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
	st.subheader("Preprocessed data")
	st.write(X[0:num])

	#Encoding categorical datas (sex, BP, Cholestrol)
	from sklearn import preprocessing
	le_sex = preprocessing.LabelEncoder()
	le_sex.fit(['F', 'M'])
	X[:, 1] = le_sex.transform(X[:, 1])

	le_BP = preprocessing.LabelEncoder()
	le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
	X[:, 2] = le_BP.transform(X[:, 2])

	le_Chol = preprocessing.LabelEncoder()
	le_Chol.fit(['NORMAL', 'HIGH'])
	X[:, 3] = le_Chol.transform(X[:, 3])
	st.subheader("Encoded data")
	st.write(X[0:num])

	st.subheader("Target variable (y)")	
	y = df["Drug"]
	st.write(y[0:num])

	#setting up decision tree
	from sklearn.model_selection import train_test_split
	X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
	st.subheader("Shape of train and test data set")
	st.write("shape of X_trainset: ", X_trainset.shape)
	st.write("shape of y_trainset: ", y_trainset.shape)
	st.write("shape of X_testset: ", X_testset.shape)
	st.write("shape of y_testset: ", y_testset.shape)

	#modeling
	drugTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
	drugTree.fit(X_trainset, y_trainset)
	predTree = drugTree.predict(X_testset)
	if st.sidebar.button("Train and predict"):
		st.write("Predicted: ",predTree[0:5])
		st.write(y_testset[0:5])
		st.write("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))	

	if st.sidebar.button("Visualize tree"):
		dot_data = StringIO()
		filename = "drugtree.png"
		featureNames = df.columns[0:5]
		targetNames = df["Drug"].unique().tolist()
		out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
		graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
		graph.write_png(filename)
		img = mpimg.imread(filename)
		plt.figure(figsize=(100, 200))
		plt.imshow(img,interpolation='nearest')
		st.pyplot()


if __name__ == '__main__':
	main()


