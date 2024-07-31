from base import (
    BaseLR,
    LogisticRegressionGD,
    LogisticRegressionSGD,
    LogisticRegressionNewton
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
        rho = 0.5,
        c = 0.5,
        backtracking = False,
        fit_intercept=True, 
        log=True
    ):
        # Check solver
        if solver not in ["gradient-descent", "newton", "sgd", "bfgs"]:
            raise ValueError("Solver must be one of ['gradient-descent', 'newton', 'sgd', 'bfgs']")
        
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
                regularization="None", # Only support no regularization
                lambda_=lambda_, 
                fit_intercept=fit_intercept, 
                log=log
            )
        elif solver == "sgd":
            self.solver = LogisticRegressionSGD(
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


    def fit(self, X, y):
        self.solver.fit(X, y)

    def predict(self, X):
        return self.solver.predict(X)
    
    def predict_prob(self, X):
        return self.solver.predict_prob(X)
