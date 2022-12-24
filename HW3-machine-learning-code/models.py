import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        weight = self.get_weights()
        return nn.DotProduct(weight, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        keepgoing = 1
        while keepgoing:
            keepgoing = 0
            for X, y in dataset.iterate_once(1):
                pred = self.get_prediction(X)
                if pred != nn.as_scalar(y):
                    keepgoing = 1
                    self.w.update(X,nn.as_scalar(y))

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.weight1 = nn.Parameter(1, 64)
        self.bias1 = nn.Parameter(1, 64)
        self.weight2 = nn.Parameter(64, 32)
        self.bias2 = nn.Parameter(1, 32)
        self.weight3 = nn.Parameter(32, 16)
        self.bias3 = nn.Parameter(1, 16)
        self.weight4 = nn.Parameter(16, 1)
        self.bias4 = nn.Parameter(1, 1)
        self.learning_rate = -0.001
        self.batch_size = 20

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        step1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1),self.bias1))
        step2 = nn.ReLU(nn.AddBias(nn.Linear(step1, self.weight2),self.bias2))
        step3 = nn.ReLU(nn.AddBias(nn.Linear(step2, self.weight3),self.bias3))
        res = nn.AddBias(nn.Linear(step3, self.weight4), self.bias4)
        return res

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        con = 1
        while con >= 0.001:
            for X, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(X, y)
                grads = nn.gradients(loss, [self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3, self.weight4, self.bias4])
                con = nn.as_scalar(loss)
                self.weight1.update(grads[0], self.learning_rate)
                self.bias1.update(grads[1], self.learning_rate)
                self.weight2.update(grads[2], self.learning_rate)
                self.bias2.update(grads[3], self.learning_rate)
                self.weight3.update(grads[4], self.learning_rate)
                self.bias3.update(grads[5], self.learning_rate)
                self.weight4.update(grads[6], self.learning_rate)
                self.bias4.update(grads[7], self.learning_rate)
