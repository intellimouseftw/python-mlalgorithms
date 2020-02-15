##################################
Python Machine Learning Algorithms
##################################

INTRODUCTION
------------
grad_descent_linear contains the code that runs the linear regression using gradient descent
algorithm to find the regression parameters.

grad_descent_logist will be used for logistic regression.


USAGE
-----
Code is applicable to different data sets, data sets will have to be manually imported and
manipulated by the user.

Model accuracy is computed after the model parameters are obtained.

The code splits the data into 3 sets, train, test and cross-validation. The split ratio can
be user defined.

Cross-validation data can be used for model accuracy comparisons against different hyper-
parameters. An example would be to tune the regularization parameter lambda.


END-NOTES
---------
These codes function similarly to using sklearn regression functions.

Possible further improvements to the code would be to incorporate K-fold validation to achieve
a better estimation of the model accuracy.
