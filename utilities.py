# Utility functions

import re

def my_replacements(text):
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