# Script to carry out analysis of the data from the Goodreads API
# Goal: predict my rating and/or positivity (or a combination of them) for unread books

import pickle
import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from collections import Counter
from utilities import pretty_cm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

with open('data/reviews_sentiment.pkl', 'r') as f:
    df = pickle.load(f)

with open('data/author_info.pkl', 'r') as f:
    df_author = pickle.load(f)

# Merge author information
df = pd.merge(df, df_author, how='left', left_on='author', right_on='author')
# Set gender to be 0/1. Nones are assumed male (0)
gender = {'male': 0, 'female': 1}
df.replace(to_replace=gender, inplace=True)
df.loc[df['gender'].isnull(), 'gender'] = 0

# Select columns to consider
cols = ['author', 'publication_year', 'average_rating', 'number_pages', 'works', 'fans', 'number_ratings', 'gender',
        'rating', 'positivity']
data = df[cols].copy()
data.dropna(axis=0, inplace=True)  # eliminate missing values
ratings = data['rating']
positivity = data['positivity']

# Get most frequent authors.
# Authors what I have read just once will be grouped together as 'Single-Read' to avoid too many variables
freq_author = Counter(data['author'])
for author in freq_author:
    if freq_author[author] < 2:
        data.replace(author, 'Single-Read Authors', inplace=True)

# Set authors as dummy variables
dummy_var = pd.get_dummies(data['author'])
data = data[cols[1:-2]]  # remove extra columns
data = data.join(dummy_var)
columns = data.columns.tolist()
data = data[columns[:7] + ['Single-Read Authors']].copy()

# Clean up unicode characters in column names
columns = data.columns.tolist()
columns = ['{}'.format(s.encode('utf-8').decode('ascii', errors='ignore')) for s in columns]

clf = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=3)
clf = clf.fit(data, ratings)

# Examine the tree
nice_columns = ['Publication Year', 'Average Rating', 'Number of Pages', 'Number of Works', 'Number of Fans',
                'Number of Ratings', 'Author Gender', 'Single-Read Authors']
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=nice_columns,
                     class_names=['{}-star'.format(x) for x in range(2, 6)],
                     filled=True, rounded=True,
                     special_characters=True, label='all')
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png("figures/decision_tree_rating.png")


# Now for the random forest
# Prepare and split data to training/test set
df_train, df_test, train_label, test_label = train_test_split(data, ratings, test_size=0.2, random_state=42)
rf = RandomForestClassifier(max_depth=None, min_samples_split=1, n_estimators=100, random_state=42)

# Fit and predict model
rf.fit(df_train, train_label)
prediction = rf.predict(df_test)
print(classification_report(test_label, prediction))
cm = confusion_matrix(test_label, prediction)
pretty_cm(cm, show_sum=True)

# Save model
joblib.dump(rf, 'data/model/ratings_model.pkl')


# Now, for positivity (as a continuous variable)
df_train, df_test, train_label, test_label = train_test_split(data, positivity, test_size=0.2, random_state=42)
rf_reg = RandomForestRegressor(max_depth=None, min_samples_split=1, n_estimators=100, random_state=42)

# Fit and predict model
rf_reg.fit(df_train, train_label)
prediction = rf_reg.predict(df_test)
diff = test_label - prediction
print('Average difference: {:.2f} and standard deviation: {:.2f}'.format(diff.mean(), diff.std()))

# Save model
joblib.dump(rf_reg, 'data/model/positivity_model.pkl')


# Feature importance
importance = pd.DataFrame(rf.feature_importances_, index=nice_columns).reset_index()
importance.columns = ['Feature', 'Importance']
importance_reg = pd.DataFrame(rf_reg.feature_importances_, index=nice_columns).reset_index()
importance_reg.columns = ['Feature', 'Importance']
importance['Type'] = ['Ratings'] * len(importance)
importance_reg['Type'] = ['Positivity'] * len(importance_reg)
importance = importance.append(importance_reg).reset_index()

g = sns.barplot(x='Feature', y='Importance', data=importance, hue='Type', palette='Set1')
g.set_xlabel('Feature')
g.set_ylabel('Importance')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
plt.tight_layout()
plt.savefig('figures/rf_features.png')
