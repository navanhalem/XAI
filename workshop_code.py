import dalex as dx
import numpy as np
import seaborn as sns

sns.set(rc={'figure.figsize': (8, 8)})

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import sample
import random
from collections import defaultdict
from sklearn.inspection import plot_partial_dependence
from lime.lime_tabular import LimeTabularExplainer

import itertools

# Load the dataset
data = fetch_california_housing(as_frame=True)

sns.scatterplot(data=data.frame, x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95),
           loc="upper left")
_ = plt.title("Median house value depending on spatial location")

sns.scatterplot(data=data.frame, x="AveRooms", y="AveBedrms")
_ = plt.title("Relation between AvgRooms en AvgBedrms")

data.frame = data.frame[(data.frame.AveRooms < 50) & (data.frame.AveBedrms < 10)]
sns.scatterplot(data=data.frame, x="AveRooms", y="AveBedrms")
_ = plt.title("Relation between AvgRooms en AvgBedrms")

sns.scatterplot(data=data.frame, x="Population", y="AveOccup")
_ = plt.title("Relation between Population en AveOccup")

data.frame = data.frame[(data.frame.Population < 17500) & (data.frame.AveOccup < 20)]
sns.scatterplot(data=data.frame, x="Population", y="AveOccup")
_ = plt.title("Relation between Population en AveOccup")

sns.histplot(data=data.frame, x="HouseAge")
_ = plt.title("Distribution of HouseAge")

sns.histplot(data=data.frame, x="MedInc")
_ = plt.title("Distribution of median income")

sns.histplot(data=data.frame, x="MedHouseVal")
_ = plt.title("Distribution of median house value")

sns.scatterplot(data=data.frame, x="Longitude", y="Latitude",
                size="MedHouseVal", hue="MedHouseVal",
                palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95),
           loc="upper left")
_ = plt.title("Median house value depending on spatial location")

# Split the data into train and test data:
# Split the data in train and test sets
np.random.seed(1)
X_train, X_test, y_train, y_test = train_test_split(data.frame[data.feature_names],
                                                    data.frame[data.target_names].iloc[:, 0], test_size=0.2)

np.random.seed(1)
# Create regression models
model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
# model = SVR().fit(X_train, y_train)
# model = MLPRegressor(hidden_layer_sizes=(16, 32, 16)).fit(X_train, y_train)

X_test_sample = pd.DataFrame(X_test.iloc[1, :]).T
y_test_sample = y_test.iloc[1]

X_test_sample
y_test_sample

exp = dx.Explainer(model, X_train, y_train)

breakdown = exp.predict_parts(X_test_sample,
                              type='break_down',
                              order=np.array(['Latitude', 'Longitude', 'AveRooms', 'AveBedrms',
                                              'Population', 'AveOccup', 'HouseAge', 'MedInc']),
                              random_state=1)
breakdown.plot()

breakdown = exp.predict_parts(X_test_sample,
                              type='break_down',
                              order=np.array(['Longitude', 'Latitude', 'AveRooms', 'AveBedrms',
                                              'Population', 'AveOccup', 'HouseAge', 'MedInc']),
                              random_state=1)
breakdown.plot()

breakdown_interaction_10 = exp.predict_parts(X_test_sample,
                                             type='break_down_interactions',
                                             interaction_preference=10,
                                             random_state=1)
breakdown_interaction_10.plot()

contributions = defaultdict(lambda: [])

random.seed(1)
for ordering in tqdm(sample(list(itertools.permutations(data.feature_names)), 100)):
    breakdown = exp.predict_parts(X_test_sample, type='break_down', order=list(ordering))
    for item in list(zip(breakdown.result.variable_name, breakdown.result.contribution))[1:-1]:
        contributions[item[0]].append(item[1])

sns.boxplot(data=pd.DataFrame(contributions))

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
lime.show_in_notebook(show_table=True)

## CP oscillations
cp_sample = exp.predict_profile(X_test_sample)
cp_sample.result
cp_sample.plot()

X_test_sample_high_medinc = X_test_sample.copy()
X_test_sample_high_medinc['MedInc'] = 7.895
model.predict(X_test_sample_high_medinc)

## CP oscillations
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

data_unif = pd.DataFrame([(k, sum(v)) for k, v in cp_oscillations_unif.items()])
data_unif.columns = ['var', 'oscillations']
sns.barplot(x="oscillations",
            y="var",
            data=data_unif.sort_values(by=['oscillations'], ascending=False, axis=0),
            color='lightblue')
_ = plt.title("cp oscillations for uniform distribution")
plt.show()

data_emp = pd.DataFrame([(k, sum(v)) for k, v in cp_oscillations_emp.items()])
data_emp.columns = ['var', 'oscillations']
sns.barplot(x="oscillations",
            y="var",
            data=data_emp.sort_values(by=['oscillations'], ascending=False, axis=0),
            color='lightblue')
_ = plt.title("cp oscillations for empirical distribution")
plt.show()

mp_rf = exp.model_parts()
mp_rf.plot()

mp_rf_grouped = exp.model_parts(variable_groups={'Location': ['Latitude', 'Longitude'],
                                                 'House': ['AveBedrms', 'AveRooms', 'HouseAge'],
                                                 'People': ['Population', 'MedInc', 'AveOccup']})
mp_rf_grouped.plot()

partial_dependence = exp.model_profile(variables=['MedInc', 'AveOccup'], N=100)
partial_dependence.plot(geom='profiles')

sns.set(rc={'figure.figsize': (16, 6)})
plot_partial_dependence(model, X_train, ['Longitude', 'Latitude', ['Longitude', 'Latitude']])

y_train_med = y_train.median()
y_train_bool = y_train.apply(lambda x: True if x > y_train_med else False)
y_test_bool = y_test.apply(lambda x: True if x > y_train_med else False)

X_test_sample = pd.DataFrame(X_test.iloc[1, :]).T
y_test_sample = y_test_bool.iloc[1]

model = RandomForestClassifier(random_state=0).fit(X_train, y_train_bool)
model.score(X_test, y_test_bool)

X_test_sample

shape = (1,) + X_train.shape[1:]
alibi_test.explainers.counterfactual.CounterFactual(model.predict_proba, shape)
