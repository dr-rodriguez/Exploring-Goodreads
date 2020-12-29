# Access and save reviews from the Goodreads API

from goodreads import client
import pandas as pd
from dateutil.parser import parse
import time
import pickle
from my_settings import api_key, api_secret, username  # loading private information

gc = client.GoodreadsClient(api_key, api_secret)
gc.authenticate()

user = gc.user(username=username)

data = list()
page = 1

# Loop over the pages grabbing all the reviews
try:
    while True:
        reviews = user.reviews(page)

        for review in reviews:
            # Eliminate non-reviews
            if review.body is None:
                continue

            # Process how long it took me to read
            # timespan = pd.to_datetime(review.read_at) - pd.to_datetime(review.started_at)  # messes up timezone
            try:
                timespan = parse(review.read_at) - parse(review.started_at)
                timespan = timespan.days
            except:
                timespan = None

            # print(review.book['title'])
            # print('{}/5'.format(review.rating))
            # print(review.body)
            # print('{} - {}'.format(review.started_at, review.read_at))
            # print('{} days'.format(timespan.days))

            data.append([review.book['title'],
                         review.book['authors']['author']['name'],
                         review.rating,
                         review.body,
                         timespan,
                         parse(review.read_at).year,
                         review.book['publication_year'],
                         review.book['publication_month'],
                         review.book['ratings_count'],
                         review.book['average_rating'],
                         review.book['num_pages']])

        page += 1
        time.sleep(1)

except KeyError: # done with reviews
    print('Could not grab additional reviews')
    pass

columns = ['title', 'author', 'rating', 'text', 'timespan', 'year_read', 'publication_year',
           'publication_month', 'number_ratings', 'average_rating', 'number_pages']

df = pd.DataFrame(data, columns=columns)

# Save to pickle to avoid re-accessing API
with open('data/reviews.pkl', 'w') as f:
    pickle.dump(df, f)

# Extract additional author information from Goodreads
author_data = list()
for name in set(df['author']):
    author = gc.find_author(name)
    print(name)
    try:
        print(author._author_dict)
        works = author.works_count
        fans = author.fans_count()['#text']
        town = author.hometown
        gender = author.gender
    except:
        works, fans, town, gender = '', '', '', ''

    author_data.append([name, works, fans, town, gender])
    time.sleep(1)

columns = ['author', 'works', 'fans', 'hometown', 'gender']
df_author = pd.DataFrame(author_data, columns=columns)
for col in ['works', 'fans']:
    df_author[col] = pd.to_numeric(df_author[col], errors='ignore')

with open('data/author_info.pkl', 'w') as f:
    pickle.dump(df_author, f)
