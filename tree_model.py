# Script to carry out analysis of the data from the Goodreads API
# Goal: predict my rating and/or positivity (or a combination of them) for unread books

import pickle
import pandas as pd
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from collections import Counter

with open('data/reviews_sentiment.pkl', 'r') as f:
    df = pickle.load(f)

# Select columns to consider
cols = ['author', 'publication_year', 'average_rating', 'number_pages', 'number_ratings', 'rating', 'positivity']
data = df[cols].copy()
data.dropna(axis=0, inplace=True)  # eliminate missing values
ratings = data['rating']
positivity = data['positivity']

# Get most frequent authors.
# Authors what I have read 1-2 times will be grouped together as 'Other' to avoid too many variables
freq_author = Counter(data['author'])
for author in freq_author:
    if freq_author[author] < 3:
        data.replace(author, 'Other Authors', inplace=True)

# Set authors as dummy variables
dummy_var = pd.get_dummies(data['author'])
data = data[cols[1:-2]]  # remove extra columns
data = data.join(dummy_var)

# Clean up unicode characters in column names
columns = data.columns.tolist()
columns = ['{}'.format(s.encode('utf-8').decode('ascii', errors='ignore')) for s in columns]

clf = tree.DecisionTreeClassifier(max_depth=7, min_samples_split=3)
clf = clf.fit(data, ratings)

# Examine the tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=columns,
                     class_names=[str(x) for x in range(2,6)],
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("figures/decision_tree_rating.pdf")

# TODO: Split data to training/test set
# TODO: Create random forest model
