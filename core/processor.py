from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
    
    def fit(self, X, y=None):
        for column in X.columns:
            le = LabelEncoder()
            le.fit(X[column])
            self.label_encoders[column] = le
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column in X.columns:
            le = self.label_encoders[column]
            X_transformed[column] = le.transform(X[column])
        return X_transformed
    
    def inverse_transform(self, X):
        X_inverse_transformed = X.copy()
        for column in X.columns:
            le = self.label_encoders[column]
            X_inverse_transformed[column] = le.inverse_transform(X[column])
        return X_inverse_transformed
