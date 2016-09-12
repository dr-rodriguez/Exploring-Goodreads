# Utility functions

import re
import pandas as pd
from collections import Counter
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import requests
import simplejson


def my_replacements(text):
    """
    Quick function to clean up some of my review text. It clears HTML and some extra characters.
    Also removes my frequent mentions of full reviews on my blog.

    :param text:
        Text to process
    :return:
        Processed text
    """

    text = re.sub(r'<(.*?)>', ' ', text)  # removing HTML code encapsulated within <>
    text = re.sub(r'\n', ' ', text)  # removing newline characters
    text = re.sub(r'&nbsp;', ' ', text)  # removing some extra HTML code
    text = re.sub(r'\"','', text)  # removing explicit quotation marks
    text = re.sub(r"\'", '', text)  # removing explicit single quotation marks

    # Text replacement
    stop_text = ["For my full review", "For a full review", "check out my blog", "Read my full review at my blog",
                "review can be found in my blog", "A full review is available on my blog", "review is up on my blog",
                 "full review", "my blog"]
    for elem in stop_text:
        text = re.sub(elem, '', text)

    return text


def get_sentiment(text, more_stop_words=['']):
    # Load up the NRC emotion lexicon
    filename = 'data/NRC-emotion-lexicon-wordlevel-alphabetized-v0.92.txt'

    # Add to stop_words
    stop_words = set(stopwords.words('english'))
    stop_words.update(more_stop_words)

    emotion_data = pd.read_csv(filename, delim_whitespace=True, skiprows=45, header=None, names=['word', 'affect', 'flag'])

    emotion_words = dict()
    emotion_map = dict()
    affects = ['positive', 'negative', 'anger', 'anticipation', 'disgust',
               'fear', 'joy', 'sadness', 'surprise', 'trust']
    for key in affects:
        emotion_words[key] = emotion_data[(emotion_data['affect'] == key) &
                                          (emotion_data['flag'] == 1)]['word'].tolist()
        emotion_map[key] = list()

    # Note no stemming or it may fail to match words
    words = Counter([i.lower() for i in wordpunct_tokenize(text)
                     if i.lower() not in stop_words and not i.lower().startswith('http')])
    for key in emotion_words.keys():
        x = set(emotion_words[key]).intersection(words.keys())
        emotion_map[key] = len(x)

    return emotion_map


def pretty_cm(cm, label_names=['3', '4', '5'], show_sum=False):
    table = pd.DataFrame(cm, columns=['P-' + s for s in label_names], index=['T-' + s for s in label_names])
    print(table)
    if show_sum:
        print('Sum of columns: {}'.format(cm.sum(axis=0)))
        print('Sum of rows: {}'.format(cm.sum(axis=1)))
    print('')


def geo_api_search(loc):
    """
    Use Google API to get location information

    :param loc: string to parse
    :return: status, latitude, longitude
    """

    url = 'http://maps.googleapis.com/maps/api/geocode/json'
    params = {'address': loc.strip(), 'sensor': 'false'}

    r = requests.get(url, params=params)
    data = simplejson.loads(r.text)

    if data['status'] in ['OK']:
        lat = data['results'][0]['geometry']['location']['lat']
        lon = data['results'][0]['geometry']['location']['lng']
        geo = [lat, lon]  # lat first, then lon
    else:
        geo = None

    return data['status'], geo

