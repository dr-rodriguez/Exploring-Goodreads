# Script to explore the data

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

# Load data extracted from API/CSV
with open('data/books.pkl', 'rb') as f:
    df = pickle.load(f)

# Remove non-reviews
df = df[~df['text'].isnull()]

# Numeric columns
numeric_columns = ['rating', 'timespan', 'year_read', 'publication_year', 'publication_month',
                   'number_ratings', 'average_rating', 'number_pages']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')


# Settings
max_words = 25
use_stemming = False
author_threshold = 3  # how many books to include for the author counts


# Word Frequency Chart of my reviews
porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
stop_words.update([s for s in string.punctuation] + ['...', 'b', '://', 'strakul', 'blogspot',
                                                     'com', 'full', 'review', 'blog', 'html', 'jpg'])
text = ''
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

word_data = pd.DataFrame([(k, top_words[k]) for k in top_words], columns=['Word', 'Counts'])
g = sns.barplot(x='Word', y='Counts', palette='rocket', data=word_data)
g.set_xlabel('Word')
g.set_ylabel('Counts')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
plt.tight_layout()
plt.savefig('figures/words_frequency.png')


# My Rating vs Avg Raging
g = sns.lmplot(x="rating", y="average_rating", data=df, x_jitter=0.5, height=8, scatter_kws={'s': 80})
g.set_xlabels('My Rating (with 0.5 jitter)')
g.set_ylabels('Average Rating')
g.savefig('figures/rating_comparison.png')


# Year read vs published
g = sns.lmplot(x="year_read", y="publication_year", hue='rating', data=df, fit_reg=False, size=8,
               legend=False, scatter_kws={'s': 80})

# Labels on old books
year_books = df[(df['publication_year'] < 1984) & (df['year_read'].notnull())]
for x, y, t in zip(year_books['year_read'].tolist(), year_books['publication_year'].tolist(),
                   year_books['title'].map(lambda x: x.split('(')[0].strip()).tolist()):
    plt.text(x+0.5, y-0.5, t, color='k', ha='center', va='top')

g.set_xlabels('Year Read (or Added)')
g.set_ylabels('Year Published')
g.ax.get_xaxis().get_major_formatter().set_useOffset(False)
g.fig.get_axes()[0].legend(title='My Rating', loc='lower right')
g.savefig('figures/years.png')


# Most frequent authors
# Get most frequent authors. Rare authors will be grouped together as 'Other' to avoid too many variables
freq_author = Counter(df['author'])
count = 0
others = 0
for author in freq_author:
    if freq_author[author] < author_threshold:
        others += freq_author[author]
    else:
        count += 1

# Quick Plot to check most frequent authors
top_authors = OrderedDict(freq_author.most_common(count))
top_authors['Other Authors'] = others
author_data = pd.DataFrame([(k, top_authors[k]) for k in top_authors], columns=['Author', 'Books'])
# g = sns.barplot(x=top_authors.keys(), y=top_authors.values(), palette="Greens_d")
g = sns.barplot(x='Author', y='Books', palette="Greens_d", data=author_data)
g.set_xlabel('Author')
g.set_ylabel('Books Reviewed')
g.set_xticklabels(g.xaxis.get_majorticklabels(), rotation=90)
plt.tight_layout()
plt.savefig('figures/author_frequency.png')
