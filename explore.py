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

# Load data extracted from API (118 reviews)
with open('reviews.pkl','r') as f:
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


g.savefig('figures/reading_rate.png')

# Shortest reads (some may be in error)
df[df['timespan'] < 2]