import numpy as np

from abc import ABC, abstractmethod
from loader import DataLoader

class BaseLR(ABC):
    def __init__(
        self,
        learning_rate=0.01,
        num_iterations=100,
        regularization="l2",
        lambda_=1.0,
        fit_intercept=True, 
        log=False
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.fit_intercept = fit_intercept
        self.log = log
        self.history = []

        # Choose the loss function and gradient function
        if self.regularization == "l1":
            self.loss = self.__loss_l1
            self.gradient = self.__gradient_l1
        elif self.regularization == "l2":
            self.loss = self.__loss_l2
            self.gradient = self.__gradient_l2
        else:
            self.loss = self.__loss
            self.gradient = self.__gradient

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y, theta=None):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def __loss_l1(self, h, y, theta):
        return self.__loss(h, y) + self.lambda_ * np.sum(np.abs(theta[1:])) / y.size
    
    def __loss_l2(self, h, y, theta):
        return self.__loss(h, y) + self.lambda_ * np.sum(np.square(theta[1:])) / (2 * y.size)
    
    def __gradient(self, h, y, X, theta=None):
        return np.dot(X.T, (h - y)) / y.size
    
    def __gradient_l1(self, h, y, X, theta):
        grad = self.__gradient(h, y, X)
        grad[1:] += self.lambda_ * np.sign(theta[1:])
        return grad
    
    def __gradient_l2(self, h, y, X, theta):
        grad = self.__gradient(h, y, X)
        grad[1:] += self.lambda_ * theta[1:]
        return grad
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()
    
    def log_loss(self, X, y, theta):
        z = np.dot(X, theta)
        h = self.__sigmoid(z)
        loss = self.__loss(h, y, theta)
        self.history.append(loss)
        return loss

    
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("Fit function is not implemented.")
    

class LogisticRegressionGD(BaseLR):
    def __init__(
        self, 
        learning_rate=0.01, 
        num_iterations=100,
        regularization="l2",
        lambda_=1.0,
        fit_intercept=True, 
        log=True, 
        rho = 0.5,
        c = 0.5,
        backtracking = False
    ):
        super().__init__(learning_rate, num_iterations, regularization, lambda_, fit_intercept, log)
        self.rho = rho
        self.c = c
        self.backtracking = backtracking

    def __find_optimal_learning_rate(self, learning_rate, rho, c, gradient, X, y):
        alpha = learning_rate
        while super()._BaseLR__loss(super()._BaseLR__sigmoid(np.dot(X, self.theta - alpha * gradient)), y) > \
              super()._BaseLR__loss(super()._BaseLR__sigmoid(np.dot(X, self.theta)), y) - c * alpha * np.linalg.norm(gradient):
            alpha *= rho

        return alpha

    def fit(self, X, y):
        if self.fit_intercept:
            X = super()._BaseLR__add_intercept(X)
        
        self.theta = np.zeros((X.shape[1], 1))
        
        for _ in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = super()._BaseLR__sigmoid(z)

            gradient = self.gradient(h, y, X, self.theta)

            if self.backtracking:
                learning_rate = self.__find_optimal_learning_rate(self.learning_rate, self.rho, self.c, gradient, X, y)
            else:
                learning_rate = self.learning_rate
            self.theta -= learning_rate * gradient
            
            if self.log == True:
                self.log_loss(X, y, self.theta)


class LogisticRegressionSGD(BaseLR):
    def __init__(
        self, 
        learning_rate=0.01, 
        num_iterations=100,
        regularization="l2",
        lambda_=1.0,
        batch_size=32,
        fit_intercept=True, 
        log=True
    ):
        super().__init__(learning_rate, num_iterations, regularization, lambda_, fit_intercept, log)
        self.batch_size = batch_size

    def fit(self, X, y):
        if self.fit_intercept:
            X = super()._BaseLR__add_intercept(X)
        
        # Create data loader
        data_loader = DataLoader(X, y, batch_size=self.batch_size)
        
        self.theta = np.zeros((X.shape[1], 1))
        
        for _ in range(self.num_iterations):
            for batch_X, batch_y in data_loader:
                z = np.dot(batch_X, self.theta)
                h = super()._BaseLR__sigmoid(z)

                gradient = self.gradient(h, batch_y, batch_X, self.theta)
                self.theta -= self.learning_rate * gradient
            
            if self.log == True:
                self.log_loss(X, y, self.theta)
    

class LogisticRegressionNewton(BaseLR):
    def __init__(
        self, 
        learning_rate=0.01, 
        num_iterations=100,
        regularization="None",
        lambda_=1.0,
        fit_intercept=True, 
        log=True
    ):
        super().__init__(learning_rate, num_iterations, regularization, lambda_, fit_intercept, log)

    def fit(self, X, y):
        if self.fit_intercept:
            X = super()._BaseLR__add_intercept(X)

        self.history = []
        
        # weights initialization
        self.theta = np.zeros((X.shape[1], 1))
        
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = super()._BaseLR__sigmoid(z)

            gradient = self.gradient(h, y, X, self.theta)
            v = (h * (1 - h)).reshape(-1, )
            hessian = np.dot(X.T, np.dot(np.diag(v), X)) / y.size
            self.theta -= self.learning_rate * np.linalg.pinv(hessian) @ gradient
            
            if self.log == True:
                self.log_loss(X, y, self.theta)
