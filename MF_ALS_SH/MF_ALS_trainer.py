import numpy as np
import matplotlib.pyplot as plt
from dataloader import dataloader as dataloader

class MF():

    def __init__(self, R, K, alpha, beta, iterations):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.K, self.num_items))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        self.W = self.R > 0.5
        self.W[self.W == True] = 1
        self.W[self.W == False] = 0
        self.W = self.W.astype(np.float64, copy=False)

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        self.errors = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.als()
            loss = self.loss()

            if i % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i+1, loss))
            self.errors.append(self.loss())
        return self.errors


    def als(self):
        """
        Perform ALS
        """
        for u in range(self.P.shape[0]):
            self.P[u,:] = np.linalg.solve(np.dot(self.Q, self.Q.T)
                                        + self.beta * np.eye(self.K),
                                          np.dot(self.Q, self.R[u,:])).T


        for i in range(self.Q.shape[0]):
            # print(self.Q.shape) 100, 9066
            # print(self.P.shape) 671, 100
            # print(self.Q[i,:].shape) 9066,
            # print(self.R[:,i].shape) 671,
            self.Q[:,i] = np.linalg.solve(np.dot(self.P.T, self.P)
                                               + self.beta * np.eye(self.K),
                                          np.dot(self.R[:,i],self.P))


    def loss(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.P.dot(self.Q)
        loss = 0
        for x, y in zip(xs, ys):
            loss += pow(self.R[x,y] - predicted[x,y ],2)

        return loss



R = dataloader()
mf = MF(R, K=100, alpha=0.1, beta=0.01, iterations=100)

errors = mf.train()
plt.plot(errors)
plt.title("Loss")
plt.show()