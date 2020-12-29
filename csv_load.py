# Given the deprecation of the Goodreads API, I instead use an Export of my library as the input
#

import pickle
import pandas as pd
from dateutil.parser import parse

df = pd.read_csv('goodreads_library_export.csv')

# Get only read books
df = df[df['Exclusive Shelf'] == 'read']

# Loop through and store certain data
data = []
for i, row in df.iterrows():

    # No longer able to get read-time since date added/date finished not exported
    timespan = None

    print(row['Title'])
    print(row['Author'])
    print('{}/5'.format(row['My Rating']))
    print('Added: {} vs Read: {}'.format(row['Date Added'], row['Date Read']))

    try:
        date_read = parse(row['Date Read']).year
    except TypeError:
        parse(row['Date Added']).year

    data.append([
        row['Title'],
        row['Author'],
        row['My Rating'],
        row['My Review'],
        None,  # no timespan
        date_read,
        row['Original Publication Year'],
        None,  # publication month
        None,  # number of ratings
        row['Average Rating'],
        row['Number of Pages']
    ])

columns = ['title', 'author', 'rating', 'text', 'timespan', 'year_read', 'publication_year',
           'publication_month', 'number_ratings', 'average_rating', 'number_pages']

df = pd.DataFrame(data, columns=columns)

# Save to pickle to avoid re-loading
with open('data/books.pkl', 'wb') as f:
    pickle.dump(df, f)
