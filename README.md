# cs231n_stanford

[lecture video](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)

[syllabus](http://cs231n.stanford.edu/syllabus.html)


## 2 Image Classification

- [course note](https://nbviewer.jupyter.org/github/mebusy/cs231n_stanford/blob/master/slider/lecture_2.pdf)
    - [2 Image classification notes](https://cs231n.github.io/classification)
    - [2 linear classification notes](http://cs231n.github.io/linear-classify)
    - [2.5 google cloud tutorial](https://github.com/cs231n/gcloud)

### K-Nearest Neighobrs

[K-Nearest Neighobrs demo](http://vision.stanford.edu/teaching/cs231n-demos/knn/)

- L1 distance: ∑|a-b|
- L2 distance: sqrt( ∑(a-b)² )
- K-Nearest Neighobrs: Instead of copying label from nearest neighbor(k=1), take **majority vote** from K closest points
- programming tips
    - To ensure that our vectorized implementation is correct, we make sure that it agrees with the naive implementation. 
    - There are many ways to decide whether two matrices are similar; one of the simplest is the Frobenius norm. 
    - In case you haven't seen it before, the Frobenius norm of two matrices is the square root of the squared sum of differences of all elements; 
    - in other words, reshape the matrices into vectors and compute the Euclidean distance between them.
    ```python
    difference = np.linalg.norm(dists - dists_one, ord='fro')
    print('One loop difference was: %f' % (difference, ))
    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')
    ```
- programming tips
    - to compute L2 distance, directly using numpy Vector operations may cause huge memory usage
    - the optimization tip is the convert ∑ (a-b)² to ∑a²+ ∑b²- ∑2ab

<details>

<summary>
differences between K-means and K-nearest neighbours
</summary>

These are completely different methods.

- **K-means** is a clustering algorithm that tries to partition a set of points into K sets (clusters) such that the points in each cluster tend to be near each other. 
    - It is unsupervised because the points have no external classification.
- **K-nearest neighbors** is a classification (or regression) algorithm that in order to determine the classification of a point, combines the classification of the K nearest points. 
    - It is supervised because you are trying to classify a point based on the known classification of other points.

</details>


### Hyperparameters

- What is the best value of k to use?
- What is the best distance to use?
- These are **hyperparameters**: choices about the algorithms themselves.

- Setting Hyperparameters
    - Split data into **train**, **val**, and **test**; 
    - choose hyperparameters on **val** 
    - and evaluate on **test**


### Linear Classifier

- Parametric Approach: Linear Classifier
    - Image → f(x,W) → n numbers giving class scores
- ![](imgs/cs231_linear_classifier_fxw.png)
- bias term
    - if your dataset is unbalanced and had many more cats than dogs for example, then the bias elements corresponding to cat would be higher than the other ones.
- ![](imgs/cs231_linear_classifier_NN.png)
-  Example with an image with 4 pixels, and 3 classes (cat/dog/ship)
    - ![](imgs/cs231_linear_classifier_fxw_ex.png)
- Linear Classifier: Three Viewpoints
    - ![](imgs/cs231_lc_3_viewpoints.png)




<details>
<summary>
Linear Regression VS Logistic Regression(binary/linear classification)
</summary>

· | Linear Regression | Logistic Regression
--- | --- | ---
outcome | continuous, e.g. y=f(x) | discrete , e.g. 1/0
data fit | is all about fitting a straight line in the data | is about fitting a curve to the data
model | regression | classification
activation function | no | need activation function to convert a linear regression equation to the logistic regression equation
estimation |  It is based on the least square estimation | It is based on maximum likelihood estimation.
application | estimate the dependent variable in case of a change in independent variables. For example, predict the price of houses. | calculate the probability of an event. For example, classify if tissue is benign or malignant.


Q: How do you explain the fact that a logistic regression model can take polynomial predictor variables, e.g. w₁·x² + w₂·x² to produce a non-linear decision boundary?  Is that still a linear classifier ?
A: The concept of "linear classifier" appears to originate with the concept of a linear mode. This polynomial example would be viewed as nonlinear in terms of the given "features" x₁ and x₂, but it would be ***linear*** in terms of features x₁² and x₂³.


</details>

- [3 Loss Function & Optimization]()
    - [3 optimization notes](https://cs231n.github.io/optimization-1/)


















