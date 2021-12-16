from numpy.matrixlib.defmatrix import matrix
import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression:
    ''' 
    Logistic regression is a fundamental classification method. It belongs to the group of linear classifiers and is somewhat similar to polynomial and linear regression. Logistic regression is fast and relatively simple, and it is convenient for you to interpret the results. Although this is essentially a binary classification method, it can also be applied to multiclass problems.
    '''
    def __init__(self):
        """
        This magic is a function that creates specific attributes of a class. In our case, these are only hyperparameters of the neural network and the values of the error function (according to the standard, it is 0, since in the future this attribute will be reassigned)
        """
        # variables for storing weights
        self.W, self.b = None, None
        # variable for storing current loss
        self.loss = None

    def accuracy(self, y, p):
        """
        This function return accuracy score between our prediction and real classes
        """
        return accuracy_score(p, y)

    def cost_function(self, p: float, y: float) -> float:
        """
        Cross-entropy is a Loss Function that can be used to quantify the difference between two Probability Distributions.
        """
    
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _init_weights(self):
        """
        This function is designed for random initialization of weights for our neural network. we will reassign the attributes of our class
        """
        # initialize normal
        self.W = np.random.randn(3, 1)
        self.b = np.random.randn()

    def sigm(self, x):
        """
        A sigmoid is a smooth monotonic increasing nonlinear function having the shape of the letter "S", which is often used to "smooth" the values of a certain value. It is used to obtain predictions of the logistic regression class.
        """
        # sigmoid (logistic) function
        return 1 / (1 + np.exp(-x))

    def forward_backward_pass(self, x: np.array, y: np.array, eta: float):
        """
        This function implements forward and backward pass and updates the parameters W / b. 
        First, we get linear predictions from our model by multiplying the matrix by the matrix of feature weights and adding b. After that, using sigmoids, we get targets for specific instances. 
        After calculating the error (cross-entropy), our forward pass is over. 
        During the reverse transfer, we calculate the derivatives of our error function with by W and b, and then update them taking into account the learning rate. 
        "passage forward and backward."
        """
        # FORWARD
        linear_pred = np.dot(x, self.W) + self.b
        y_pred = self.sigm(linear_pred)
        # FORWARD ENDS

        # calculate loss
        self.loss = self.cost_function(y_pred, y)

        # BACKWARD
        # here you need to calculate all the derivatives of loss with respect to W and b

        dLdW = (y_pred - y) * x.T
        dLdb = (y_pred - y)

        # then update W and b
        # don't forget the learning rate - 'eta'!
       
        self.W = self.W - eta * dLdW
        self.b = self.b - eta * dLdb

        # BACKWARD ENDS

    def fit(self, X: np.array, Y: np.array, eta=0.01, decay=0.999, iters=1000) ->list:
        """
        This function is designed to train our logistic regression. First we initialize random weights. 
        Iteratively (depending on the number of epochs) we will perform forward-backward-pass, changing hyperparameters and learning rate. 
        We also add the error we calculated to the buffer for future graph output (as our model learns)
        """
        self._init_weights()

        # buffer - for printing out mean loss over 100 iterations
        buffer = []
        # L - history of losses
        L = []

        # perform iterative gradient descent
        for i in range(iters):
            index = np.random.randint(0, len(X))
            x = X[index]
            y = Y[index]
            # update params
            self.forward_backward_pass(x, y, eta)
            # update learning rate
            eta *= decay

            L.append(self.loss)
            buffer.append(self.loss)

        return L

    def predict(self, x: np.array) -> np.array:
        """
         The function is designed to predict the class of a particular instance based on its attributes.
        """
        # Note you have to return actual classes (not probs)
        linear_pred = np.dot(x, self.W) + self.b
        y_pred = self.sigm(linear_pred)
        return np.round(y_pred)