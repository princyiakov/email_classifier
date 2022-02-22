import string
import datetime
import email
import os
import argparse

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score

from utils import nltk, get_text_from_email, clean_data


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Your Working Directory is : ", current_dir)
    parser = argparse.ArgumentParser(description='Email Theme Classifier')
    parser.add_argument('--input_file', type=str,
                        default=os.path.join(current_dir, 'giskard_dataset.csv'), required=False,
                        help='location of the input file')
    args = parser.parse_args()

    emails_df = pd.read_csv(args.input_file, sep=';')

    # Parse the emails into a list email objects
    messages = list(map(email.message_from_string, emails_df['Message']))
    emails_df.drop('Message', axis=1, inplace=True)

    # Get fields from parsed email objects
    keys = messages[0].keys()
    for key in keys:
        emails_df[key] = [doc[key] for doc in messages]

    # Parse content from emails
    emails_df['Content'] = list(map(get_text_from_email, messages))
    emails_df = emails_df[['X-Folder', 'Subject', 'Content', 'Date', 'Target']]
    emails_df = emails_df.rename(columns={'X-Folder': 'Folder'})

    # To Confirm there are no blank data
    emails_df = emails_df.drop(
        emails_df.query('Date == "" | Subject == "" | Content == "" | Folder ==""').index)
    emails_df.reset_index(drop=True, inplace=True)  # To reset the index of the dropped rows

    # Concat folder Subject and Content
    emails_df['Text'] = emails_df['Folder'] + " " + emails_df[
        'Subject'] + " " + emails_df['Content']

    emails_df['Text'] = emails_df.apply(lambda x: clean_data(x['Text'], 'lem'),
                                        axis=1)
    text_df = emails_df[['Text', 'Target']]
    num_df = emails_df[['Date']]

    le = preprocessing.LabelEncoder()
    text_df['Target'] = le.fit_transform(text_df.Target)

if __name__ == '__main__':
    main()
