import email

import pandas as pd
from utils import get_day_part, get_text_from_email


class EmailClassifier:
    def __init__(self, dataframe, text_flag=False):
        self.df = dataframe
        self.text_flag = text_flag

    def transform_df(self):
        """
        Function to transform the data into either text only columns or text and numeric values
        flag = True : Convert to text values only
        flag = False : Convert to text and numeric values
        """
        # Parse the emails into a list email objects
        messages = list(map(email.message_from_string, self.df['Message']))
        self.df.drop('Message', axis=1, inplace=True)

        # Get fields from parsed email objects
        keys = messages[0].keys()
        for key in keys:
            self.df[key] = [doc[key] for doc in messages]

        # Parse content from emails
        self.df['Content'] = list(map(get_text_from_email, messages))
        self.df = self.df.rename(columns={'X-Folder': 'Folder'})

        self.df = self.df.drop(
            self.df.query('Date == "" | Subject == "" | Content == "" | Folder ==""').index)
        self.df.reset_index(drop=True, inplace=True)  # To reset the index of the dropped rows

        self.df['Folder'] = self.df['Folder'].apply(lambda x: x.split('\\')[1])
        self.df['Date'] = pd.to_datetime(self.df['Date'], infer_datetime_format=True, utc=True)
        if self.text_flag:
            self.df['Date'] = self.df.Date.dt.day_name() + " " + self.df.Date.dt.hour.apply(
                get_day_part) + "" + self.df.Date.dt.month_name()
            self.df['Text'] = self.df['Date'] + " " + self.df['Folder'] + " " + self.df[
                'Subject'] + " " + self.df['Content']
            self.df = self.df[['Text', 'Target']].copy()
        else:
            self.df['Text'] = self.df['Folder'] + " " + self.df['Subject'] + " " + self.df[
                'Content']
            self.df['Month'] = self.df.Date.dt.month
            self.df['Day'] = self.df.Date.dt.day
            self.df['Hour'] = self.df.Date.dt.hour
            self.df = self.df[['Day', 'Month', 'Hour', 'Text', 'Target']].copy()
        return self.df
