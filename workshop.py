import itertools
import random
from collections import defaultdict
from random import sample

import dalex as dx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import plot_partial_dependence
from sklearn.model_selection import train_test_split
from tqdm import tqdm

np.random.seed(1)
sns.set(rc={'figure.figsize': (8, 8)})

# Load the dataset
data = fetch_california_housing(as_frame=True)

# Plot the mean house value per block with relation to the geographical location of the block
sns.scatterplot(data=data.frame, x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95),
           loc="upper left")
_ = plt.title("Median house value depending on spatial location")

# Plot the average rooms values against the average bedrooms values
sns.scatterplot(data=data.frame, x="AveRooms", y="AveBedrms")
_ = plt.title("Relation between AvgRooms en AvgBedrms")

# Correct the average rooms and the average bedrooms values
# Plot the corrected average rooms values against the corrected average bedrooms values
data.frame = data.frame[(data.frame.AveRooms < 50) & (data.frame.AveBedrms < 10)]
sns.scatterplot(data=data.frame, x="AveRooms", y="AveBedrms")
_ = plt.title("Relation between AvgRooms en AvgBedrms")

# Plot the population values against the average occupation values
sns.scatterplot(data=data.frame, x="Population", y="AveOccup")
_ = plt.title("Relation between Population en AveOccup")

# Correct the population and the average occupation values
# Plot the corrected population values against the corrected average occupation values
data.frame = data.frame[(data.frame.Population < 17500) & (data.frame.AveOccup < 20)]
sns.scatterplot(data=data.frame, x="Population", y="AveOccup")
_ = plt.title("Relation between Population en AveOccup")

# Plot the distribution of house age
sns.histplot(data=data.frame, x="HouseAge")
_ = plt.title("Distribution of HouseAge")

# Plot the distribution of median income
sns.histplot(data=data.frame, x="MedInc")
_ = plt.title("Distribution of median income")

# Plot the distribution of median house value
sns.histplot(data=data.frame, x="MedHouseVal")
_ = plt.title("Distribution of median house value")

# Plot the mean house value per block with relation to the geographical location of the block after corrections
sns.scatterplot(data=data.frame, x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95),
           loc="upper left")
_ = plt.title("Median house value depending on spatial location")

# Split the data in train and test sets
np.random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(data.frame[data.feature_names],
                                                    data.frame[data.target_names].iloc[:, 0], test_size=0.2)

# Create regression models
np.random.seed(1)
model = RandomForestRegressor(random_state=0).fit(X_train, y_train)

# Take a sample from the test data
X_test_sample = pd.DataFrame(X_test.iloc[1, :]).T
y_test_sample = y_test.iloc[1]
print({X_test_sample})
print({y_test_sample})

# Create the explainer object
exp = dx.Explainer(model, X_train, y_train)

# Breakdown plot with latitude first, then longitude
breakdown = exp.predict_parts(X_test_sample,
                              type='break_down',
                              order=np.array(['Latitude', 'Longitude', 'AveRooms', 'AveBedrms',
                                              'Population', 'AveOccup', 'HouseAge', 'MedInc']),
                              random_state=1)
breakdown.plot()

# Breakdown plot with longitude first, then latitude
breakdown = exp.predict_parts(X_test_sample,
                              type='break_down',
                              order=np.array(['Longitude', 'Latitude', 'AveRooms', 'AveBedrms',
                                              'Population', 'AveOccup', 'HouseAge', 'MedInc']),
                              random_state=1)
breakdown.plot()

# Breakdown plot with interactions
breakdown_interaction_10 = exp.predict_parts(X_test_sample,
                                             type='break_down_interactions',
                                             interaction_preference=10,
                                             random_state=1)
breakdown_interaction_10.plot()

# Create n random breakdown plots (in order to show the differences)
n = 50
random.seed(1)
contributions = defaultdict(lambda: [])
for ordering in tqdm(sample(list(itertools.permutations(data.feature_names)), 100)):
    breakdown = exp.predict_parts(X_test_sample, type='break_down', order=list(ordering))
    for item in list(zip(breakdown.result.variable_name, breakdown.result.contribution))[1:-1]:
        contributions[item[0]].append(item[1])

sns.boxplot(data=pd.DataFrame(contributions))
_ = plt.title(f'Contribution values for different variables, N={n}')

# Calculate the Shapley values and plot them
shapley_values = exp.predict_parts(X_test_sample, type='shap', random_state=1)
shapley_values.plot()

explainer = LimeTabularExplainer(X_train,
                                 mode="regression",
                                 feature_names=X_train.columns,
                                 discretize_continuous=False,
                                 verbose=True,
                                 random_state=1,
                                 kernel_width=1)
lime = explainer.explain_instance(X_test_sample.iloc[0, :], model.predict)
lime.as_pyplot_figure()
# lime.show_in_notebook(show_table=True)

# Create CP profile for the sample
cp_sample = exp.predict_profile(X_test_sample)
print(cp_sample.result)
cp_sample.plot()

# Manually change the MedInc value of the sample; the prediction now corresponds to that indicated by the CP profile
X_test_sample_high_medinc = X_test_sample.copy()
X_test_sample_high_medinc['MedInc'] = 7.895
model.predict(X_test_sample_high_medinc)

# Calculate the CP oscillations profiles, for uniform sampling and sampling from the empirical distribution
prediction = model.predict(X_test_sample)
cp_sample_res = exp.predict_profile(X_test_sample).result

cp_oscillations_unif = {}
cp_oscillations_emp = {}
for feature in data.feature_names:
    feature_sublist = cp_sample_res[cp_sample_res._vname_ == feature]
    feature_value_list = feature_sublist[feature].values
    cp_oscillations = (feature_sublist._yhat_ - prediction).values
    cp_oscillations_abs = np.abs(cp_oscillations)
    step_size = feature_value_list[-1] - feature_value_list[-2]
    emp_corr = [len(X_train[(feature_value - step_size * 0.5 < X_train[feature]) &
                            (X_train[feature] < feature_value + step_size * 0.5)]) / len(X_train)
                for feature_value in feature_value_list]
    cp_oscillations_unif[feature] = cp_oscillations_abs / len(feature_sublist)
    cp_oscillations_emp[feature] = cp_oscillations_abs * emp_corr

# Plot CP oscillation values for uniform distribution
data_unif = pd.DataFrame([(k, sum(v)) for k, v in cp_oscillations_unif.items()])
data_unif.columns = ['var', 'oscillations']
sns.barplot(x="oscillations", y="var", data=data_unif.sort_values(by=['oscillations'], ascending=False, axis=0), color='lightblue')
_ = plt.title("cp oscillations for uniform distribution")
plt.show()

# Plot CP oscillation values for empirical distribution
data_emp = pd.DataFrame([(k, sum(v)) for k, v in cp_oscillations_emp.items()])
data_emp.columns = ['var', 'oscillations']
sns.barplot(x="oscillations", y="var", data=data_emp.sort_values(by=['oscillations'], ascending=False, axis=0), color='lightblue')
_ = plt.title("cp oscillations for empirical distribution")
plt.show()

# Calculate and plot variable importance
mp_rf = exp.model_parts()
mp_rf.plot()

# Calculate and plot grouped variable importance
mp_rf_grouped = exp.model_parts(variable_groups={'Location': ['Latitude', 'Longitude'],
                                                 'House': ['AveBedrms', 'AveRooms', 'HouseAge'],
                                                 'People': ['Population', 'MedInc', 'AveOccup']})
mp_rf_grouped.plot()

# Plot partial dependence profile plots
partial_dependence = exp.model_profile(variables=['MedInc', 'AveOccup'], N=100)
partial_dependence.plot(geom='profiles')

sns.set(rc={'figure.figsize': (16, 6)})
plot_partial_dependence(model, X_train, ['Longitude', 'Latitude', ['Longitude', 'Latitude']])
