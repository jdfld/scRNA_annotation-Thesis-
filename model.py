import sklearn as sk
import numpy as np

class model:
    def __init__(self,model_type='rfc',params=None):
        if model_type == 'rfc':
            if params:
                self.model = sk.ensemble.RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'])
            else:
                self.model = sk.ensemble.RandomForestClassifier()
            self.encoder = sk.preprocessing.OneHotEncoder(handle_unknown='ignore')

    def predict(self,X):
        return self.model.predict(X)

    def predict_acc(self,X,y):
        y = self.encoder.transform(y)
        if y is not np.array:
            y = y.toarray()
        pred = self.model.predict(X)
        return sk.metrics.accuracy_score(y_true=y,y_pred=self.predict(X))

    def fit(self,X,y):
        y = self.encoder.fit_transform(y)
        if y is not np.array:
            y = y.toarray()
        self.model.fit(X,y)
        return sk.metrics.accuracy_score(y_true=y,y_pred=self.predict(X))
