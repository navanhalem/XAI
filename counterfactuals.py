import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from alibi.explainers.counterfactual import CounterFactual
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(1)
sns.set(rc={'figure.figsize': (8, 8)})
tf.compat.v1.disable_eager_execution()

# Load the dataset and organise data
data = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.frame[data.feature_names],
                                                    data.frame[data.target_names].iloc[:, 0], test_size=0.2)
y_train_med = y_train.median()
y_train_bool = y_train.apply(lambda x: True if x > y_train_med else False)
y_test_bool = y_test.apply(lambda x: True if x > y_train_med else False)

# Select a sample from the data
X_test_sample = pd.DataFrame(X_test.iloc[1, :]).T
y_test_sample = y_test_bool.iloc[1]

# Train (and evaluate) a classifier
model = RandomForestClassifier(random_state=0).fit(X_train, y_train_bool)
model.score(X_test, y_test_bool)

# Customise the sample and make a prediction for it
X_test_sample_high = X_test_sample.copy()
X_test_sample_high.MedInc = 4.2
model.predict(X_test_sample_high)

# Create a CounterFactual object, and generate an explanation for it
counterfactual = CounterFactual(predict_fn=model.predict_proba, shape=X_test_sample_high.shape)
explanation = counterfactual.explain(np.array(X_test_sample_high))

# Create a dataframe containing the counterfactual,
# and a scaled version of it with respect to the original
cf_df_original = pd.DataFrame(explanation.cf['X'])
cf_df_original.columns = X_test_sample_high.columns
cf_df_scaled = pd.DataFrame(explanation.cf['X'] / X_test_sample_high.values)
cf_df_scaled.columns = X_test_sample_high.columns

# Compare the original values with the counterfactual
print(X_test_sample_high)
print(cf_df_original)

# Create a dataframe denoting the change in the original values
# that needs to take place in order to get a different prediction
change = pd.DataFrame([(c, v) for c, v in zip(cf_df_scaled.values[0], cf_df_scaled.columns)])
change.columns = ['Change', 'Variable']

# Create a visual of this change
sns.barplot(x="Variable", y="Change", data=change.sort_values(by=['Variable'], ascending=False, axis=0),
            color='lightblue')
_ = plt.title("Change needed to get a counterfactual example")
plt.show()
