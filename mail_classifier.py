import os
import argparse
import pickle

import pandas as pd

from model import EmailClassifier


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

    emails_df = pd.read_csv(args.input_file, sep=';')

    preprocess = EmailClassifier(emails_df)
    emails_df = preprocess.transform_df()
    emails_df = emails_df.to_numpy()
    prediction = model.predict(emails_df)

    themes = ["company image -- current", "alliances / partnerships"
        , "california energy crisis / california politics"
        , "company image -- changing / influencing", "internal company operations"
        , "internal company policy ", "internal projects -- progress and strategy", "legal advice"
        , "meeting minutes", "political influence / contributions / contacts"
        , "regulations and regulators (includes price caps) ", "talking points", "trip reports"]

    print("Mail belongs to : ", themes[int(prediction)])


if __name__ == '__main__':
    main()
