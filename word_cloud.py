# Word Cloud of my reviews

import pickle
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.misc import imread
import matplotlib.pyplot as plt
from utilities import my_replacements

# Load data extracted from API (118 reviews)
with open('reviews.pkl','r') as f:
    df = pickle.load(f)

filename = 'figures/wordcloud.png'
size = 10
max_words = 200
horizontal = 0.9
image = 'images/book_clipart3.png'
mask = 'images/book_clipart3.png'

words = ' '.join(df['text'].tolist())

# Clean up the text
words = my_replacements(words)

# Remove URLs, 'RT' text, screen names, etc
# my_stopwords = ['RT', 'amp', 'lt']
# words_no_urls = ' '.join([word for word in words.split()
#                           if 'http' not in word and word not in my_stopwords and not word.startswith('@')
#                           ])
words_no_urls = ' '.join([word for word in words.split() if 'http' not in word])

# Add stopwords, if needed
stopwords = STOPWORDS.copy()
stopwords.add("RT")
stopwords.add('amp')
stopwords.add('lt')

# Load up a logo as a mask & color image
cloud_mask = imread(mask)
logo = imread(image)

# Generate colors
image_colors = ImageColorGenerator(logo)

# Generate plot
wc = WordCloud(stopwords=stopwords, mask=cloud_mask, color_func=image_colors, scale=0.8,
               max_words=max_words, background_color='white', random_state=42, prefer_horizontal=horizontal)
# wc = WordCloud(stopwords=stopwords, scale=0.8, max_words=max_words, background_color='white',
#                random_state=42, prefer_horizontal=horizontal)

wc.generate(words_no_urls)

plt.figure(figsize=(size, size))
plt.imshow(wc)
plt.axis("off")
plt.savefig(filename)