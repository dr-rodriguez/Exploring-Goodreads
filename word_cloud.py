# Word Cloud of my reviews

import pickle
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from utilities import my_replacements

# Load data extracted from API (118 reviews)
with open('data/books.pkl', 'rb') as f:
    df = pickle.load(f)

# Remove non-reviews
df = df[~df['text'].isnull()]

filename = 'figures/wordcloud.png'
size = 10
max_words = 200
horizontal = 0.9
image = 'images/book_clipart3.png'
mask = 'images/book_clipart3.png'

words = ''
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
logo = np.array(Image.open(image))

# Generate colors
image_colors = ImageColorGenerator(logo)

# Generate plot
wc = WordCloud(stopwords=stopwords, mask=logo, color_func=image_colors, scale=0.8,
               max_words=max_words, background_color='white', random_state=42, prefer_horizontal=horizontal)

wc.generate(words_no_urls)

plt.figure(figsize=(size, size))
plt.imshow(wc)
plt.axis("off")
plt.savefig(filename)
