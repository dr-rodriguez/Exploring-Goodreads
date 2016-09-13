# Apply the models to predict what ratings I will give for books on my To-Read list

from goodreads import client
import pandas as pd
import time
import pickle
from my_settings import api_key, api_secret, username  # loading private information
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

gc = client.GoodreadsClient(api_key, api_secret)
user = gc.user(username=username)

# Get shelf
page = 1
data = list()
try:
    while True:
        r = gc.request("/review/list/{}.xml".format(user.gid), {'shelf': 'to-read', 'page': page}).get('books')

        for book in r['book']:
            data.append([book['title'],
                         book['authors']['author']['name'],
                         book['publication_year'],
                         book['average_rating'],
                         book['num_pages'],
                         book['ratings_count']])

        page += 1
        time.sleep(1)

except KeyError:  # done with reviews
    print('Could not grab additional reviews')
    pass

columns = ['title', 'author', 'publication_year', 'average_rating', 'number_pages', 'number_ratings']
df_shelf = pd.DataFrame(data, columns=columns)

# Populate with author information
author_data = list()
for name in set(df_shelf['author']):
    author = gc.find_author(name)
    print(name)
    try:
        print author._author_dict
        works = author.works_count
        fans = author.fans_count()['#text']
        gender = author.gender
    except:
        works, fans, gender = '', '', ''

    author_data.append([name, works, fans, gender])
    time.sleep(1)

columns = ['author', 'works', 'fans', 'gender']
df_author = pd.DataFrame(author_data, columns=columns)

# Merge with to-read books
df_shelf = pd.merge(df_shelf, df_author, how='left', left_on='author', right_on='author')

# Save for future use
with open('data/to-read_shelf.pkl', 'w') as f:
    pickle.dump(df_shelf, f)

# ==========================================================
# Alternative: load up existing shelf
with open('data/to-read_shelf.pkl', 'r') as f:
    df_shelf = pickle.load(f)

# Load up my prior reviews
with open('data/reviews.pkl', 'r') as f:
    reviews = pickle.load(f)

# Process authors
data = df_shelf.copy()
read_authors = set(reviews['author'])
for author in set(data['author']):
    if author in read_authors:
        # print '{}: READ'.format(author.encode('utf-8').decode('ascii', errors='ignore'))
        data.replace(author, 0, inplace=True)
    else:
        # print '{}: NEW'.format(author.encode('utf-8').decode('ascii', errors='ignore'))
        data.replace(author, 1, inplace=True)

# Set gender to be 0/1. Nones are assumed male (0)
gender = {'male': 0, 'female': 1}
data.replace(to_replace=gender, inplace=True)
data.loc[data['gender'].isnull(), 'gender'] = 0

cols = ['author', 'publication_year', 'average_rating', 'number_pages', 'works', 'fans', 'number_ratings', 'gender']
data = data[cols].copy()
data.columns = ['Single-Read Authors'] + cols[1:]
# data.dropna(axis=0, inplace=True)  # eliminate missing values
# Impute missing values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0).fit(data)
data = pd.DataFrame(imp.transform(data), columns=data.columns)

# Load up models and apply them
rf = joblib.load('data/model/ratings_model.pkl')
rf_reg = joblib.load('data/model/positivity_model.pkl')
rating = rf.predict(data)
positivity = rf_reg.predict(data)

# Examine predictions
data['rating'] = rating
data['positivity'] = positivity
final = df_shelf[['title', 'author']].join(data).dropna(axis=0)
final.sort_values(by=['rating', 'average_rating'], axis=0, ascending=False).head(5)
final.sort_values(by=['positivity', 'average_rating'], axis=0, ascending=False).head(5)

# Intersection of top-rated, top-positive
pos_10 = final['positivity'].sort_values(ascending=False)[int(len(final)*0.1)+1]  # 10%
best = final[(final['rating'] == 2) & (final['positivity'] > pos_10)]  # 1 if binary 5-star/not, 2 if 5/4/less, or 5
cols = ['title', 'author', 'average_rating', 'positivity']
best['title'] = best['title'].apply(lambda x: x.split('(')[0].strip())  # eliminate series designation. Gives a warning
final_best = best[cols].sort_values(by=['positivity', 'average_rating'], axis=0, ascending=False)
nice_cols = ['Title', 'Author', 'Average Rating', 'Positivity']
final_best.columns = nice_cols

print(final_best)

final_best.to_html()