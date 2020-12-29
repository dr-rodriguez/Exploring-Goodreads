# Exploration of page counts

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load data extracted from API/CSV
with open('data/books.pkl', 'rb') as f:
    df = pickle.load(f)

# Plot year and pages
g = sns.boxplot(x="year_read", y="number_pages", data=df, palette="vlag", whis=[0, 100], width=.6)
# Add in points
g = sns.stripplot(x="year_read", y="number_pages", data=df, size=4, color=".3", linewidth=0)
g.set_xlabel('Year Read (or Added)')
g.set_ylabel('Number of Pages')
plt.tight_layout()
plt.savefig('figures/page_histogram.png')


# Column for Yes/No if Reviewed
df = df.assign(Reviewed=np.where(df.text.isnull(), 'No', 'Yes'))

# Plot number of books per year, divided by whether I review them or not
g = sns.countplot(x='year_read', data=df, palette="deep", hue='Reviewed')
g = sns.countplot(x='year_read', data=df, facecolor=(0, 0, 0, 0), edgecolor='black')  # add outline for clarity and total counts
g.set_xlabel('Year Read (or Added)')
g.set_ylabel('Number of Books')
plt.tight_layout()
plt.savefig('figures/book_counts.png')
