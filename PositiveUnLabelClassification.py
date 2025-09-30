import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.tree import DecisionTreeClassifier
n_estimators = 1000
estimator = DecisionTreeClassifier()

X, y = make_circles(
    n_samples=6000, noise=0.1,
    shuffle = True, factor=0.65
)

x = pd.DataFrame(X, columns= ['feature1', 'feature2'])
y = pd.Series(y)

hidden_size = 1000

# Keep the original targets safe for later
y_orig = y.copy()
y.loc[
    np.random.choice(
        y[y == 1].index,
        replace = False,
        size = hidden_size
    )
] = 0

# positive index
iP = y[y > 0].index
# unlabeled index
iU = y[y <= 0].index

num_oob = pd.DataFrame(np.zeros(shape = y.shape), index = y.index)
sum_oob = pd.DataFrame(np.zeros(shape = y.shape), index = y.index)

for _ in range(n_estimators):
    # Get a bootstrap sample of unlabeled points for this round
    ib = np.random.choice(iU, replace = True, size = len(iP))
    # find the oob data points for this round
    i_oob = list(set(iU) - set(ib))

    # Get the training data (ALL positives and the bootstrap
    # sample of unlabeled points) and build the tree
    Xb = X[y > 0].append(X.loc[ib])
    yb = y[y > 0].append(y.loc[ib])
    estimator.fit(Xb, yb)

    # Record the OOB scores from this round
    sum_oob.loc[i_oob, 0] += estimator.predict_proba(X.loc[i_oob])[:,1]
    num_oob.loc[i_oob, 0] += 1
