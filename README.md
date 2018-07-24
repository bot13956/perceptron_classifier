# perceptron_classifier
This code applies the perceptron classification algorithm to the iris data set.The weights used for computing the activation function are calculated using the least-square method.This method is different from Rosenblatt's original perceptron rule where the weights are calculated recursively.

For more information about the implementation of Rosenblatt's perceptron algorithm, see the following book:"Python Machine Learning" by Sebastian Raschka.

Created on Tue Jul 24 12:36:55 2018
@author: Benjamin Tayo

iris.data.csv: contains the iris dataset obtained from: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
The dataset contains the following Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica

perceptron.ipynb: jupyter notebook that implements the least-square perceptron fit algorithm

perceptron.py: corresponding python script 
