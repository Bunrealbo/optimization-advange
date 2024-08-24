from .base import (
    BaseLR,
    LogisticRegressionGD,
    LogisticRegressionBatchGD,
    LogisticRegressionNewton,
    LogisticRegressionBFGS,
    LogisticRegressionAdam,
    LogisticRegressionAcceleration
)

class LogisticRegression:
    def __init__(
        self, 
        learning_rate=0.01, 
        num_iterations=100,
        solver="gradient-descent",
        regularization="None",
        lambda_=1.0,
        batch_size=32,
        rho=0.5,
        c=0.5,
        backtracking = False,
        fit_intercept=True, 
        log=True,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        # Check solver
        __SUPPORTED_SOLVERS = ["gradient-descent", "newton", "batch-gd", "bfgs", "adam", "accelerated-gd"] 
        if solver not in __SUPPORTED_SOLVERS:
            raise ValueError(f"Solver must be one of {__SUPPORTED_SOLVERS}")
        
        if solver == "gradient-descent":
            self.solver = LogisticRegressionGD(
                learning_rate=learning_rate, 
                num_iterations=num_iterations, 
                regularization=regularization, 
                lambda_=lambda_, 
                backtracking=backtracking,
                rho=rho,
                c=c,
                fit_intercept=fit_intercept, 
                log=log
            )
        elif solver == "newton":
            self.solver = LogisticRegressionNewton(
                learning_rate=learning_rate, 
                num_iterations=num_iterations, 
                regularization=regularization,
                lambda_=lambda_, 
                fit_intercept=fit_intercept, 
                log=log
            )
        elif solver == "batch-gd":
            self.solver = LogisticRegressionBatchGD(
                learning_rate=learning_rate, 
                num_iterations=num_iterations, 
                regularization=regularization, 
                lambda_=lambda_, 
                batch_size=batch_size, 
                fit_intercept=fit_intercept, 
                log=log
            )
        elif solver == "bfgs":
            self.solver = LogisticRegressionBFGS(
                learning_rate=learning_rate, 
                num_iterations=num_iterations, 
                regularization=regularization, 
                lambda_=lambda_, 
                fit_intercept=fit_intercept, 
                log=log            
            )
        elif solver == "adam":
            self.solver = LogisticRegressionAdam(
                learning_rate=learning_rate, 
                num_iterations=num_iterations, 
                regularization=regularization, 
                lambda_=lambda_, 
                fit_intercept=fit_intercept, 
                log=log,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon
            )
        elif solver == "accelerated-gd":
            self.solver = LogisticRegressionAcceleration(
                learning_rate=learning_rate, 
                num_iterations=num_iterations, 
                regularization=regularization, 
                lambda_=lambda_, 
                fit_intercept=fit_intercept, 
                log=log
            )

    def fit(self, X, y):
        self.solver.fit(X, y)

    def predict(self, X):
        return self.solver.predict(X)
    
    def predict_prob(self, X):
        return self.solver.predict_prob(X)
