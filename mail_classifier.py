import string
import datetime
import email
import os
import argparse
import pickle

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score

from utils import nltk, get_text_from_email, clean_data, transform_date


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Your Working Directory is : ", current_dir)
    parser = argparse.ArgumentParser(description='Email Theme Classifier')
    parser.add_argument('--model_file', type=str,
                        default=os.path.join(current_dir, 'gbdt_model.pkl'), required=False,
                        help='location of the model file')
    parser.add_argument('--input_file', type=str,
                        default=os.path.join(current_dir, 'test_dataset.csv'), required=True,
                        help='location of the input file')
    args = parser.parse_args()

    emails_df = pd.read_csv(args.input_file, sep=';')

    # Load Model
    model = pickle.load(open(args.model_file, 'rb'))

    # Parse the emails into a list email objects
    messages = list(map(email.message_from_string, emails_df['Message']))
    emails_df.drop('Message', axis=1, inplace=True)

    # Get fields from parsed email objects
    keys = messages[0].keys()
    for key in keys:
        emails_df[key] = [doc[key] for doc in messages]

    # Parse content from emails
    emails_df['Content'] = list(map(get_text_from_email, messages))
    emails_df = emails_df[['X-Folder', 'Subject', 'Content', 'Date']]
    emails_df = emails_df.rename(columns={'X-Folder': 'Folder'})

    # Concat folder Subject and Content
    emails_df['Text'] = emails_df['Folder'] + " " + emails_df[
        'Subject'] + " " + emails_df['Content']

    emails_df['Text'] = emails_df.apply(lambda x: clean_data(x['Text'], 'lem'),
                                        axis=1)
    text_df = emails_df[['Text']]
    v = TfidfVectorizer(max_features=3000)
    X = v.fit_transform(text_df['Text'])
    transformedtext_df = pd.DataFrame(X.toarray(), columns=v.get_feature_names())

    num_df = transform_date(emails_df[['Date']])

    df = pd.concat([transformedtext_df, num_df], axis=1)
    print(df)
    prediction = model.predict(df)
    output = int(prediction[0])


if __name__ == '__main__':
    main()
