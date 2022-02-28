import email
import os
import argparse
import pickle

import pandas as pd

from utils import get_text_from_email, clean_data, get_day_part
from sklearn.feature_extraction.text import TfidfVectorizer



def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print("Your Working Directory is : ", current_dir)
    parser = argparse.ArgumentParser(description='Email Theme Classifier')
    parser.add_argument('--model_file', type=str,
                        default=os.path.join(current_dir, 'svc_model.pkl'), required=False,
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

    emails_df['Date'] = pd.to_datetime(emails_df['Date'], infer_datetime_format=True, utc=True)
    emails_df['day_part'] = emails_df.Date.dt.day_name() + " " + emails_df.Date.dt.hour.apply(
        get_day_part) + " " + emails_df.Date.dt.month_name()

    emails_df['Folder'] = emails_df['Folder'].apply(lambda x: x.split('\\')[1])

    emails_df['Text'] = emails_df['day_part'] + " " + emails_df['Folder'] + " " + emails_df[
        'Subject'] + " " + emails_df['Content']

    emails_df['Text'] = emails_df.apply(lambda x: clean_data(x['Text'], 'lem'), axis=1)


    prediction = model.predict(emails_df['Text'])
    output = int(prediction[0])

    themes = {
        0: "company image -- current"
        , 1: "alliances / partnerships"
        , 2: "california energy crisis / california politics"
        , 3: "company image -- changing / influencing"
        , 4: "internal company operations"
        , 5: "internal company policy"
        , 6: "internal projects -- progress and strategy"
        , 7: "legal advice"
        , 8: "meeting minutes"
        , 9: "political influence / contributions / contacts"
        , 10: "regulations and regulators (includes price caps)"
        , 11: "talking points"
        , 12: "trip reports"
    }
    print("Mail belongs to : ", themes[output])


if __name__ == '__main__':
    main()
