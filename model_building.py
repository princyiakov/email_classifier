import os
import argparse
import pickle

import pandas as pd

from model import EmailClassifier
from utils import get_text_data, get_numeric_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

current_dir = os.path.dirname(os.path.abspath(__file__))
print("Your Working Directory is : ", current_dir)

parser = argparse.ArgumentParser(description='Model Building for Email Theme Classifier')
parser.add_argument('--input_file', type=str,
                    default=os.path.join(current_dir, 'giskard_dataset.csv'), required=False,
                    help='location of the input file placed in the working directory')
args = parser.parse_args()

emails_df = pd.read_csv(args.input_file, sep=';')

model = EmailClassifier(emails_df, train_flag=True)
emails_df = model.transform_df()

X = emails_df.iloc[:, :4]
y = emails_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.to_numpy()

transformer_numeric = FunctionTransformer(get_numeric_data)
transformer_text = FunctionTransformer(get_text_data)

# Create a pipeline to concatenate Tfidf Vector and Numeric data
# Use RandomForestClassifier as an example
pipeline = Pipeline([
    ('features', FeatureUnion([
        ('numeric_features', Pipeline([
            ('selector', transformer_numeric)
        ])),
        ('text_features', Pipeline([
            ('selector', transformer_text),
            ('vec', TfidfVectorizer(analyzer='word'))
        ]))
    ])),
    ('scaler', StandardScaler(with_mean=False)),
    ('clf', SVC())
])

svc_model = pipeline.fit(X_train, y_train)

# open a file, where you ant to store the data
file = open('svc_model.pkl', 'wb')

# dump information to that file
pickle.dump(svc_model, file)
