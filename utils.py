import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

nltk.download('omw-1.4')


def get_text_from_email(message):
    """
    To get the content from email objects
    """
    parts = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    return ''.join(parts)


def clean_data(columns, clean=None):
    """
    To Clean Data from columns of dfs
    Removes unnecessary character, lowercases words
    and performs stemming/lemitizing
    """
    data = re.sub(r"=[\s]{1}", "",
                  columns)  # To remove unwanted charater with patter "=" + number
    data = re.sub(r"=[\d]{1,4}", "",
                  data)  # To remove unwanted charater with patter "=" + number
    data = re.sub(r"&", "", data)  # To remove =
    data = re.sub(r"\S*https?:\S*", "", data)  # To remove URL
    data = re.sub(r"\S*.com\S*", "", data)  # To remove websites
    data = re.sub(r"[^a-zA-Z0-9]", " ", data)
    data = re.sub(r'\s+', ' ', data).strip()
    data = data.lower()
    data = nltk.word_tokenize(data)
    stopwords = nltk.corpus.stopwords.words('english')
    newstopwords = ['ect', 'hou', 'com', 'recipient', 'cc', 'na', 'ees', 'th', 'pm', 'folder',
                    'folders', 'pst']
    stopwords.extend(newstopwords)

    if clean == 'stem':
        ps = PorterStemmer()
        data = [ps.stem(word) for word in data if
                word not in stopwords]  # stemming all words that are not stopwords
    elif clean == 'lem':
        lm = WordNetLemmatizer()  # initialize lemmatizing
        data = [lm.lemmatize(word) for word in data if
                word not in stopwords]  # lemmitizing all words that are not stopwords

    else:
        data = [word for word in data if
                word not in stopwords]  # stemming all words that are not stopwords

    data = ' '.join(data)

    return data


def get_day_part(x):
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return 'Noon'
    elif (x > 16) and (x <= 20):
        return 'Evening'
    elif (x > 20) and (x <= 24):
        return 'Night'
    elif x <= 4:
        return 'Late Night'


# Create Function Transformer to use Feature Union
def get_numeric_data(x):
    return [record[:-2].astype(float) for record in x]

def get_text_data(x):
    return [record[-1] for record in x]
