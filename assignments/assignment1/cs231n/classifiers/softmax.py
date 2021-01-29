from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax( f, aggregate_axis = 0 ):
    # instead: first shift the values of f so that the highest number is 0:
    f_max = np.max(f, axis= aggregate_axis )
    f -= f_max.reshape( f_max.shape + (1,) )
    f_exp = np.exp(f)
    f_sum = np.sum( f_exp, axis=aggregate_axis )
    return f_exp / f_sum.reshape( f_sum.shape + (1,) )


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        x = X[i]
        scores = x.dot( W )
        scores_softmax = softmax( scores )  

        loss += - np.log( scores_softmax [ y[i] ] )
        
        for j in range( num_classes ):
            dW[:, j] += scores_softmax[ j ] * x 
        dW[:, y[i] ] -= x

    loss /= num_train
    loss += reg * (W * W).sum()

    dW /= num_train
    dW += reg * 2 * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = X.dot( W )
    scores_softmax = softmax( scores, aggregate_axis=1 )  # N,C

    loss = np.sum( - np.log( scores_softmax [ np.arange( num_train ), y ] ) )

    loss /= num_train
    loss += reg * (W * W).sum()
    
    scores_softmax[ np.arange( num_train ), y ] -= 1
    dW = X.T.dot( scores_softmax )
    
    dW /= num_train
    dW += reg * 2 * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
