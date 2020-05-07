# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import argparse


class Dataset:

    def __init__(self, input_filepath, args):
        self.df = pd.read_csv(input_filepath)
        self.savepath = args.savepath
        self.charlimit = args.charlimit

    def output_filename(self, name):
        """Simple function to quickly fetch directory path and return the
        right filepath for saving files.
        Also creates the save folder if it doesn't already exist"""
        dirname = os.path.dirname(__file__)
        Path(os.path.join(dirname, self.savepath + name)
             ).mkdir(parents=True, exist_ok=True)
        return os.path.join(dirname, self.savepath + name)

    def clean(self, df):
        """Performs data cleaning (removes unnecessary symbols, new line characters and null records)"""

        data = df.copy()
        data['upvotes'] = pd.to_numeric(data['upvotes'], errors='coerce')
        data.loc[:, 'comment_body'] = data['comment_body'].str.replace('\([http]{4}[a-zA-Z0-9.:/]+\)','(url)',regex=True).replace('x200B', '').replace(r'[^0-9a-zA-Z.>\'-:!=?/(), ]+', ' ', regex=True)
        return data

    def preprocess(self, df, labels, charlimit, onehotencoding=False):
        """
        Preprocesses the cleaned version of the data (ordinal encoding/one hot encoding)
        Labels parameter to set naming conventions for the categories (In our case, upvotes are divided into
        {<=0: poor, 1-3: Normal, 4-9: Good, 9+: Best})

        Size parameter to get smaller version of the dataset (for faster training)
        Char limit parameter for filtering out text exceeding length limit (ex: removing comments larger than 512 characters)

        Use onehotencoding parameter to use One hot enconding (True) or ordinal encoding (False)"""
        data = df.copy()
        data = data.loc[data['comment_body'].apply(
            lambda x: len(x) <= charlimit), :]
        data["label"] = pd.DataFrame(
            pd.cut(data.loc[:, "upvotes"], bins=[-np.inf, 0., 3, 9, np.inf], labels=labels))
        data.rename(columns={"comment_body": "text"}, inplace=True)
        if onehotencoding:
            label_cols = pd.get_dummies(data["label"])
            data = data.drop(columns=['label', 'upvotes'])
            data = pd.concat([data, label_cols], axis=1)
        else:
            data = data.drop(columns=['upvotes'])
        return data.reset_index(drop=True)

    def savedata(self, df, outputfolder, val=None):
        """Shuffles and splits the dataset into training, test and validation sets, for use in various models.
        Saves the files in appropriate folders. Takes the dataframe and output/save paths to train, test and validation sets
        as parameters."""

        df = shuffle(df, random_state=42).reset_index(drop=True)
        splitby = int(0.8*len(df))
        train_data = df[:splitby]
        if val:
            splitby1 = int(0.9*len(df))
            test_data = df[splitby:splitby1]
            val_data = df[splitby1:]
            val_data = val_data.dropna().reset_index(drop=True)
            val_data.to_csv(f"{outputfolder}/val.csv", index=False)
        else:
            test_data = df[splitby:]
        train_data = train_data.dropna().reset_index(drop=True)
        test_data = test_data.dropna().reset_index(drop=True)
        train_data.to_csv(f"{outputfolder}/train.csv", index=False)
        test_data.to_csv(f"{outputfolder}/test.csv", index=False)

    def collect(self):
        """Main function that calls data cleaning, preprocessing and saving functions"""

        labels = ['poor', 'normal', 'good', 'best']
        flairlabels = ['__label__'+x for x in labels]
        data = self.df.copy()
        data = self.clean(data)
        bertdata = self.preprocess(data, [0, 1, 2, 3], self.charlimit)
        fastdata = self.preprocess(data, labels, self.charlimit)
        flairdata = self.preprocess(data, flairlabels, self.charlimit)
        self.savedata(bertdata, self.output_filename('bertdata/'))
        self.savedata(fastdata, self.output_filename('fastdata/'))
        self.savedata(flairdata, self.output_filename('flairdata/'), True)


def build_parser():
    parser = argparse.ArgumentParser()
    CHARLIMIT = 512
    OUTPUTFOLDER = "../../data/processed/"
    parser.add_argument('--savepath',
                        dest='savepath', help='Path to store the output files folder',
                        metavar='Output data folder', default=OUTPUTFOLDER)
    parser.add_argument('--charlimit', type=int,
                        dest='charlimit', help='Remove comments larger than (default %(default)s) characters',
                        default=CHARLIMIT)
    return parser


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../../data/raw/data.csv')
    parser = build_parser()
    args = parser.parse_args()
    Dataset(filename, args).collect()


if __name__ == '__main__':
    main()
