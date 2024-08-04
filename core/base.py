import numpy as np

from abc import ABC, abstractmethod
from scipy import sparse as sp
from loader import DataLoader
from time import time

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
        self.times = []

        # Choose the loss function and gradient function
        if self.regularization == "l1":
            self.loss = self.__loss_l1
            self.gradient = self.__gradient_l1
            self.hessian = self.__hessian
        elif self.regularization == "l2":
            self.loss = self.__loss_l2
            self.gradient = self.__gradient_l2
            self.hessian = self.__hessian_l2
        else:
            self.loss = self.__loss
            self.gradient = self.__gradient
            self.hessian = self.__hessian

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y, theta=None):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def __loss_l1(self, h, y, theta):
        return self.__loss(h, y) + self.lambda_ * np.sum(np.abs(theta[1:]))
    
    def __loss_l2(self, h, y, theta):
        return self.__loss(h, y) + self.lambda_ * np.sum(np.square(theta[1:])) / 2
    
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
    
    def __hessian(self, h, X):
        v = (h * (1 - h)).reshape(-1, )
        return X.T @ sp.diags(v) @ X / h.size
    
    def __hessian_l2(self, h, X):
        return self.__hessian(h, X) + self.lambda_ * np.eye(X.shape[1])
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()
    
    def logging_loss(self, X, y, theta):
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
              super()._BaseLR__loss(super()._BaseLR__sigmoid(np.dot(X, self.theta)), y) - c * alpha * (np.linalg.norm(gradient)**2):
            alpha *= rho

        return alpha

    def fit(self, X, y):
        if self.fit_intercept:
            X = super()._BaseLR__add_intercept(X)
        
        self.theta = np.zeros((X.shape[1], 1))
        
        start = time()
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
                self.logging_loss(X, y, self.theta)
                self.times.append(time() - start)


# Proximal Gradient Descent with Acceleration (often referred to as Fast Iterative Shrinkage-Thresholding Algorithm or FISTA)
class LogisticRegressionPA(BaseLR): 
    def __init__(
        self,
        learning_rate=0.01,
        num_iterations=100,
        regularization="l1",
        lambda_=1.0,
        fit_intercept=True,
        log=True,
    ):
        super().__init__(learning_rate, num_iterations, regularization, lambda_, fit_intercept, log)

        if self.regularization == "l1":
            self.gradient = super()._BaseLR__gradient # Use the default gradient function for g(x)
        
    def soft_threshold(self, theta, lambda_):
        return np.sign(theta) * np.maximum(np.abs(theta) - lambda_, 0)

    def fit(self, X, y):
        if self.fit_intercept:
            X = super()._BaseLR__add_intercept(X)
        
        self.theta = np.zeros((X.shape[1], 1))
        theta_previous = self.theta.copy()

        start = time()
        for i in range(1, self.num_iterations + 1):
            z = np.dot(X, self.theta)
            h = super()._BaseLR__sigmoid(z)
            v = self.theta + ((i - 2) / (i + 1)) * (self.theta - theta_previous)
            gradient = self.gradient(h, y, X, self.theta)
            theta_previous = self.theta.copy()

            if self.regularization == "l1": # Proximal operator for L1 regularization
                # Update theta
                self.theta = self.soft_threshold(v - self.learning_rate * gradient, self.lambda_ * self.learning_rate)

            else: # Otherwise, use the default gradient descent
                # Update theta
                self.theta = v - self.learning_rate * gradient

            if self.log == True:
                self.logging_loss(X, y, self.theta)
                self.times.append(time() - start)


class LogisticRegressionBatchGD(BaseLR):
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
        
        self.step_history = []
        self.theta = np.zeros((X.shape[1], 1))
        
        start = time()
        for _ in range(self.num_iterations):
            total_loss = 0

            # Create data loader
            data_loader = DataLoader(X, y, batch_size=self.batch_size)
            
            for batch_X, batch_y in data_loader:
                z = np.dot(batch_X, self.theta)
                h = super()._BaseLR__sigmoid(z)

                gradient = self.gradient(h, batch_y, batch_X, self.theta)
                self.theta -= self.learning_rate * gradient

                z = np.dot(batch_X, self.theta)
                h = super()._BaseLR__sigmoid(z)
                loss = super()._BaseLR__loss(h, batch_y, self.theta)
                total_loss += loss
                self.step_history.append(loss)
            
            if self.log == True:
                self.history.append(total_loss / len(data_loader))
                self.times.append(time() - start)
    

class LogisticRegressionNewton(BaseLR):
    def __init__(
        self, 
        learning_rate=0.01, 
        num_iterations=100,
        regularization="None",
        lambda_=1.0,
        fit_intercept=True, 
        log=True,
        rho=0.5,
        c=0.5,
        backtracking=False
    ):
        super().__init__(learning_rate, num_iterations, regularization, lambda_, fit_intercept, log)
        self.rho = rho
        self.c = c
        self.backtracking = backtracking

    def __find_optimal_learning_rate(self, learning_rate, rho, c, gradient, hessian, X, y):
        alpha = learning_rate
        p_k = - np.linalg.pinv(hessian) @ gradient # descent direction
        while super()._BaseLR__loss(super()._BaseLR__sigmoid(np.dot(X, self.theta + alpha * p_k)), y) > \
              super()._BaseLR__loss(super()._BaseLR__sigmoid(np.dot(X, self.theta)), y) + c * alpha * gradient.T @ p_k:
            alpha *= rho

        return alpha

    def fit(self, X, y):
        if self.fit_intercept:
            X = super()._BaseLR__add_intercept(X)

        self.history = []
        
        # weights initialization
        self.theta = np.zeros((X.shape[1], 1))
        
        start = time()
        for i in range(self.num_iterations):
            z = np.dot(X, self.theta)
            h = super()._BaseLR__sigmoid(z)

            gradient = self.gradient(h, y, X, self.theta)
            v = (h * (1 - h)).reshape(-1, )
            hessian = self.hessian(h, X)

            if self.backtracking:
                learning_rate = self.__find_optimal_learning_rate(self.learning_rate, self.rho, self.c, gradient, hessian, X, y)
            else:
                learning_rate = self.learning_rate
            self.theta -= learning_rate * np.linalg.pinv(hessian) @ gradient
            
            if self.log == True:
                self.logging_loss(X, y, self.theta)
                self.times.append(time() - start)


class LogisticRegressionBFGS(BaseLR):
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
        
        self.theta = np.zeros((X.shape[1], 1))
        z = np.dot(X, self.theta)
        h = super()._BaseLR__sigmoid(z)
        gradient = self.gradient(h, y, X, self.theta)

        H = np.eye(X.shape[1])
        
        start = time()
        for _ in range(self.num_iterations):
            p = -H @ gradient
            s = self.learning_rate * p
            theta_new = self.theta + s
            
            z_new = np.dot(X, theta_new)
            h_new = super()._BaseLR__sigmoid(z_new)
            gradient_new = self.gradient(h_new, y, X, theta_new)
            
            delta_gradient = gradient_new - gradient
            
            r = 1/(delta_gradient.T@s)
            
            li = (np.eye(X.shape[1])-(r*((s@(delta_gradient.T)))))
            ri = (np.eye(X.shape[1])-(r*((delta_gradient@(s.T)))))
            
            H = li @ H @ ri + (r*((s@(s.T))))
            
            gradient = gradient_new
            self.theta = theta_new
            
            z = np.dot(X, self.theta)
            h = super()._BaseLR__sigmoid(z)
            (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        
            if self.log == True:
                self.logging_loss(X, y, self.theta)
                self.times.append(time() - start)


class LogisticRegressionAdam(BaseLR):
    def __init__(
        self,
        learning_rate=0.01,
        num_iterations=100,
        regularization="None",
        lambda_=1.0,
        fit_intercept=True,
        log=True,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        super().__init__(learning_rate, num_iterations, regularization, lambda_, fit_intercept, log)
        self.beta1 = beta1
        self.beta2 = beta2        
        self.epsilon = epsilon
        
    def fit(self, X, y):
        if self.fit_intercept:
            X = super()._BaseLR__add_intercept(X)
        
        self.theta = np.zeros((X.shape[1], 1))
        m = np.zeros((X.shape[1], 1))
        v = np.zeros((X.shape[1], 1))
        
        start = time()
        for t in range(1, self.num_iterations+1):
            z = np.dot(X, self.theta)
            h = super()._BaseLR__sigmoid(z)
            gradient = self.gradient(h, y, X, self.theta)
        
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * np.multiply(gradient, gradient)
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            
            self.theta = self.theta - np.divide(self.learning_rate*m_hat, (np.sqrt(v_hat) + self.epsilon))

            if self.log == True:
                self.logging_loss(X, y, self.theta)
                self.times.append(time() - start)
