import numpy as np
import seaborn as sns

sns.set(rc={'figure.figsize': (8, 8)})

from alibi.explainers.counterfactual import CounterFactual
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = fetch_california_housing(as_frame=True)

np.random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(data.frame[data.feature_names],
                                                    data.frame[data.target_names].iloc[:, 0], test_size=0.2)

y_train_med = y_train.median()
y_train_bool = y_train.apply(lambda x: True if x > y_train_med else False)
y_test_bool = y_test.apply(lambda x: True if x > y_train_med else False)

X_test_sample = pd.DataFrame(X_test.iloc[1, :]).T
y_test_sample = y_test_bool.iloc[1]

model = RandomForestClassifier(random_state=0).fit(X_train, y_train_bool)
model.score(X_test, y_test_bool)

X_test_sample_high = X_test_sample.copy()
X_test_sample_high.MedInc = 4.2
model.predict(X_test_sample_high)

shape = (1,) + X_train.shape[1:]
cf = CounterFactual(model.predict_proba, shape)
expl = cf.explain(np.array(X_test_sample_high))
cf_df = pd.DataFrame(expl.cf['X'] / X_test_sample_high.values)
cf_df.columns = X_test_sample_high.columns

change = pd.DataFrame([(c, v) for c, v in zip(cf_df.values[0], cf_df.columns)])
change.columns = ['change of value', 'change']
sns.barplot(x="change", y="change of value", data=change.sort_values(by=['change'], ascending=False, axis=0), color='lightblue')
_ = plt.title("change to get a cf example")
