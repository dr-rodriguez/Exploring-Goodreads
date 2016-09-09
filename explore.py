# Script to explore the data obtained from the Google API

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
import string
from utilities import my_replacements
from collections import OrderedDict
import numpy as np

# Load data extracted from API (118 reviews)
with open('data/reviews.pkl','r') as f:
    df = pickle.load(f)

# Numeric columns
numeric_columns = ['rating', 'timespan', 'year_read', 'publication_year', 'publication_month',
                   'number_ratings', 'average_rating', 'number_pages']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')


# Settings
max_words = 20
use_stemming = False


# Word Frequency Chart of my reviews
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.update([s for s in string.punctuation] + ['...', 'b', '://', 'strakul', 'blogspot',
                                                     'com', 'full', 'review', 'blog'])

text = ' '.join(df['text'].tolist())

# Clean up the text
text = my_replacements(text)

if use_stemming:
    words = Counter([porter.stem(i.lower()) for i in wordpunct_tokenize(text)
                 if i.lower() not in stop_words and not i.lower().startswith('http')])
else:
    words = Counter([i.lower() for i in wordpunct_tokenize(text)
                     if i.lower() not in stop_words and not i.lower().startswith('http')])


top_words = OrderedDict(words.most_common(max_words))

g = sns.barplot(x=top_words.keys(), y=top_words.values(), palette='Blues')
g.set_xlabel('Word')
g.set_ylabel('Counts')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
plt.tight_layout()
plt.savefig('figures/words_frequency.png')


# My Rating vs Avg Raging
g = sns.lmplot(x="rating", y="average_rating", data=df, x_jitter=0.5, size=8, scatter_kws={'s': 80})
g.set_xlabels('My Rating (with 0.5 jitter)')
g.set_ylabels('Average Rating')
g.savefig('figures/rating_comparison.png')


# Year read vs published
g = sns.lmplot(x="year_read", y="publication_year", hue='rating', data=df, fit_reg=False, size=8,
               legend=False, scatter_kws={'s': 80})

# Labels on old books
year_books = df[(df['publication_year'] < 1995) & (df['year_read'].notnull())]
for x, y, t in zip(year_books['year_read'].tolist(), year_books['publication_year'].tolist(),
                   year_books['title'].map(lambda x: x.split('(')[0].strip()).tolist()):
    plt.text(x+0.5, y-0.5, t, color='k', ha='center', va='top')

g.set_xlabels('Year Read')
g.set_ylabels('Year Published')
g.ax.get_xaxis().get_major_formatter().set_useOffset(False)
g.fig.get_axes()[0].legend(title= 'My Rating', loc='lower right')
g.savefig('figures/years.png')


# Days to read vs Number of pages
g = sns.lmplot(x="number_pages", y="timespan", data=df, size=8, scatter_kws={'s': 80})
g = (g.set(xlim=(-10, 1250), ylim=(-10, 120))
     .set_xlabels('Number of Pages')
     .set_ylabels('Days to Read'))
# Labels on longest to read
longest = df[(df['timespan'] > 30) & (df['number_pages'].notnull())]
for x, y, t in zip(longest['number_pages'].tolist(), longest['timespan'].tolist(),
                   longest['title'].map(lambda x: x.split('(')[0].strip()).tolist()):
    plt.text(x-1, y+2, t, color='k', ha='right', va='center')

g.savefig('figures/reading_rate.png')


# Shortest reads (some may be in error)
df[df['timespan'] < 2]


# Reading rate over time
df['rate'] = df['number_pages']/df['timespan']
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Fixing infinite values
df.groupby(by='year_read', axis=0)[['timespan','number_pages','rate']].mean()

g = sns.factorplot(x='year_read', y='rate', data=df, size=8)
g = (g.set(ylim=(-10, 200))
     .set_xlabels('Year Read')
     .set_ylabels('Number of Pages Read Per Day'))
# Annotate with how many books read for that year. NOTE: not all books have pages/timespan
count = df.groupby(by='year_read', axis=0)[['rate', 'title']].count().reset_index()
for i, row in count.iterrows():
    plt.text(i, 2, '{}/{}'.format(row['rate'], row['title']), color='k', ha='center', va='center')
g.savefig('figures/reading_rate_2.png')


# Most frequent authors
# Get most frequent authors. Rare authors will be grouped together as 'Other' to avoid too many variables
freq_author = Counter(df['author'])
count = 0
others = 0
for author in freq_author:
    if freq_author[author] < 2:
        others += freq_author[author]
    else:
        count += 1

# Quick Plot to check most frequent authors
top_authors = OrderedDict(freq_author.most_common(count))
top_authors['Other Authors'] = others
g = sns.barplot(x=top_authors.keys(), y=top_authors.values(), palette="Greens_d")
g.set_xlabel('Author')
g.set_ylabel('Books Read')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
plt.tight_layout()
plt.savefig('figures/author_frequency.png')
