# Sentiment analysis of my reviews

import pickle
import pandas as pd
import seaborn as sns
from utilities import my_replacements, get_sentiment
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load data extracted from API (118 reviews)
with open('data/reviews.pkl','r') as f:
    df = pickle.load(f)

# Numeric columns
numeric_columns = ['rating', 'timespan', 'year_read', 'publication_year', 'publication_month',
                   'number_ratings', 'average_rating', 'number_pages']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='ignore')

# text = ' '.join(df['text'].tolist())

# Clean up the text & get sentiment
sentiment = list()
with tqdm(total=len(df)) as pbar:
    for elem in df['text'].tolist():
        new_elem = my_replacements(elem)
        elem_emotion = get_sentiment(new_elem)
        sentiment.append(elem_emotion)
        pbar.update(1)

# import textwrap
# print(textwrap.fill(new_elem))

df_sentiment = pd.DataFrame(sentiment)
df = df.join(df_sentiment)

# Plots
df['positivity'] = df['positive'] - df['negative']

g = sns.lmplot(x='average_rating', y='positivity', data=df, y_jitter=0.5, size=8, scatter_kws={'s': 80})
g = (g.set_xlabels('Average Rating')
     .set_ylabels('Positivity (with 0.5 jitter)'))
# Labels on best
subset = df[(df['positivity'] > 10) & (df['average_rating'].notnull())]
for x, y, t in zip(subset['average_rating'].tolist(), subset['positivity'].tolist(),
                   subset['title'].map(lambda x: x.split('(')[0].strip()).tolist()):
    plt.text(x-0.01, y, t, color='k', ha='right', va='bottom')

g.savefig('figures/sentiment.png')

# With *my* rating
g = sns.lmplot(x='rating', y='positivity', data=df, x_jitter=0.5, y_jitter=0.5, size=8, scatter_kws={'s': 80})
g = (g.set_xlabels('My Rating (with 0.5 jitter)')
     .set_ylabels('Positivity (with 0.5 jitter)'))

g.savefig('figures/sentiment_2.png')

# With color-coding
g = sns.lmplot(x='average_rating', y='positivity', hue='rating', data=df, y_jitter=0.5, size=8, scatter_kws={'s': 80},
               legend=False, fit_reg=False)
g = (g.set_xlabels('Average Rating')
     .set_ylabels('Positivity (with 0.5 jitter)')
     .fig.get_axes()[0].legend(title= 'My Rating', loc='lower right'))
# Labels on best
subset = df[((df['average_rating'] < 3.3) | (df['positivity'] > 10)) & (df['average_rating'].notnull())]
for x, y, t in zip(subset['average_rating'].tolist(), subset['positivity'].tolist(),
                   subset['title'].map(lambda x: x.split('(')[0].strip()).tolist()):
    plt.text(x, y, t, color='k', ha='left', va='bottom')

plt.savefig('figures/sentiment_3.png')