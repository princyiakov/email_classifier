import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import pandas as pd

nltk.download('omw-1.4')


def get_text_from_email(msg):
    '''
    To get the content from email objects
    '''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    return ''.join(parts)


def clean_data(columns, clean=None):
    """
    To Clean Data from columns of dataframes
    Removes unnecessary character, lowercases words
    and performs stemming/lemitizing
    """
    columns = re.sub("=\n", "", columns)  # To remove the '=' at every EOL
    columns = re.sub(r"&", "", columns)  # To remove the '&' in company names
    columns = re.sub(r"U.S.", "US", columns)  # To remove the '&' in company names
    stopwords = nltk.corpus.stopwords.words('english')
    newstopwords = ['ect', 'hou', 'com', 'recipient', 'cc', 'na', 'ees', 'th', 'pm']
    stopwords.extend(newstopwords)
    data = re.sub(r"\S*https?:\S*", "", columns)
    data = re.sub('[^a-zA-Z]', ' ', columns)  # removes all characters except a-z and A-Z
    data = data.lower()
    data = nltk.word_tokenize(data)

    if clean == 'stem':
        ps = PorterStemmer()
        data = [ps.stem(word) for word in data if
                not word in (stopwords)]  # stemming all words that are not stopwords
    elif clean == 'lem':
        lm = WordNetLemmatizer()  # initialize lemmatizing
        data = [lm.lemmatize(word) for word in data if
                not word in (stopwords)]  # lemmitizing all words that are not stopwords

    else:
        data = [word for word in data if
                not word in (stopwords)]  # stemming all words that are not stopwords

    data = ' '.join(data)
    # data = ' '.join(i for i in data if i not in (string.punctuation))

    return data


def transform_date(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'], infer_datetime_format=True, utc=True)
    dataframe.index = pd.to_datetime(dataframe.Date)
    dataframe['Month'] = dataframe.Date.dt.month
    dataframe['Day'] = dataframe.Date.dt.day
    dataframe['Hour'] = dataframe.Date.dt.hour
    dataframe['Timeslot'] = 0

    # pre working early morning hours
    dataframe.loc[dataframe['Date'].between_time('00:00', '08:00'), 'Timeslot'] = 0
    # working morning hours
    dataframe.loc[dataframe['Date'].between_time('08:01', '12:00'), 'Timeslot'] = 1
    # working afternoon hours
    dataframe.loc[dataframe['Date'].between_time('12:01', '18:00'), 'Timeslot'] = 2
    # post working hours to midnight
    dataframe.loc[dataframe['Date'].between_time('18:01', '23:59'), 'Timeslot'] = 3

    dataframe.drop('Date', inplace=True, axis=1)
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe