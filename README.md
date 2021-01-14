# cs231n_stanford

[lecture video](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)

[syllabus](http://cs231n.stanford.edu/syllabus.html)


- []
    - [2 Image classification notes](https://cs231n.github.io/classification)
    - [2 linear classification notes](http://cs231n.github.io/linear-classify)
    - [2.5 google cloud tutorial](https://github.com/cs231n/gcloud)
- [3 Loss Function & Optimization]()
    - [3 optimization notes](https://cs231n.github.io/optimization-1/)





## K-Nearest Neighobrs

[K-Nearest Neighobrs](http://vision.stanford.edu/teaching/cs231n-demos/knn/)


### differences between K-means and K-nearest neighbours

These are completely different methods.

- **K-means** is a clustering algorithm that tries to partition a set of points into K sets (clusters) such that the points in each cluster tend to be near each other. 
    - It is unsupervised because the points have no external classification.
- **K-nearest neighbors** is a classification (or regression) algorithm that in order to determine the classification of a point, combines the classification of the K nearest points. 
    - It is supervised because you are trying to classify a point based on the known classification of other points.

### Setting Hyperparameters

- Split data into **train**, **val**, and **test**; 
    - choose hyperparameters on **val** 
    - and evaluate on **test**


### Linear Regression VS Logistic Regression

· | Linear Regression | Logistic Regression
--- | --- | ---
outcome | continuous, e.g. y=f(x) | discrete , e.g. 1/0
data fit | is all about fitting a straight line in the data | is about fitting a curve to the data
model | regression | classification
activation function | no | need activation function to convert a linear regression equation to the logistic regression equation
estimation |  It is based on the least square estimation | It is based on maximum likelihood estimation.
application | estimate the dependent variable in case of a change in independent variables. For example, predict the price of houses. | calculate the probability of an event. For example, classify if tissue is benign or malignant.







