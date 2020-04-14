# -*- coding: utf-8 -*-
import logging, os
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.utils import shuffle

class Dataset:

    def __init__(self, input_filepath):
        self.df = pd.read_csv(input_filepath)
        self.dirname = os.path.dirname(__file__)
        os.makedirs(os.path.join(self.dirname, "../../data/processed"), exist_ok=True)
        os.makedirs(os.path.join(self.dirname, "../../data/processed/bertdata"), exist_ok=True)
        os.makedirs(os.path.join(self.dirname, "../../data/processed/fastdata"), exist_ok=True)
        os.makedirs(os.path.join(self.dirname, "../../data/processed/flairdata"), exist_ok=True)

    def output_filename(self, name):
        """Simple function to quickly fetch directory path and return the
        right filepath for saving files"""
        return os.path.join(self.dirname, "../../data/processed/" + name)

    def clean(self, data):
        """Performs data cleaning (removes unnecessary symbols, new line characters and null records)"""

        data['upvotes'] = pd.to_numeric(data['upvotes'], errors='coerce')
        data.loc[:, 'comment_body'] = data['comment_body'].replace('>“¨«»®´·º½¾¿¡§£₤‘’', '').replace(r'\[\S+\]', '', regex=True).replace(r'\(\S+\)', '', regex=True).replace(r'[http]{4}\S+', '', regex=True).replace('(\n|\r)+', '', regex=True)
        data = data.loc[data['comment_body'].apply(lambda x: x != ''), :]
        data = data.dropna()
        return data

    def preprocess(self, df, labels, size, charlimit, onehotencoding=False):
        """
        Preprocesses the cleaned version of the data (ordinal encoding/one hot encoding)
        Labels parameter to set naming conventions for the categories (In our case, upvotes are divided into
        {<=0: poor, 1-3: Normal, 4-9: Good, 9+: Best})

        Size parameter to get smaller version of the dataset (for faster training)
        Char limit parameter for filtering out text exceeding length limit (ex: removing comments larger than 512 characters)

        Use onehotencoding parameter to use One hot enconding (True) or ordinal encoding (False)"""
        data = df.copy()
        data = data.loc[data['comment_body'].apply(lambda x: len(x) <= charlimit), :]
        data["label"] = pd.DataFrame(pd.cut(data.loc[:, "upvotes"], bins=[-np.inf, 0., 3, 9, np.inf], labels=labels))
        data.rename(columns={"comment_body": "text"}, inplace=True)
        if onehotencoding:
            label_cols = pd.get_dummies(data["label"])
            data = data.drop(columns=['label', 'upvotes'])
            data = pd.concat([data, label_cols], axis=1)
        else:
            data = data.drop(columns=['upvotes'])
        return data.reset_index(drop=True)


    def savedata(self, df, trainpath, testpath, flag=None):
        """Shuffles and splits the dataset into training, test and validation sets, for use in various models.
        Saves the files in appropriate folders."""

        df = shuffle(df, random_state=42).reset_index(drop=True)
        splitby = int(0.8*len(df))
        train_data = df[:splitby]
        if flag:
            splitby1 = int(0.9*len(df))
            test_data = df[splitby:splitby1]
            val_data = df[splitby1:]
            val_data = val_data.dropna().reset_index(drop=True)
            val_data.to_csv(self.output_filename("flairdata/val.csv"), index=False)
        else:
            test_data = df[splitby:]
        train_data = train_data.dropna().reset_index(drop=True)
        test_data = test_data.dropna().reset_index(drop=True)
        train_data.to_csv(trainpath, index=False)
        test_data.to_csv(testpath, index=False)    


    def collect(self):
        """Main function that calls data cleaning, preprocessing and saving functions"""

        labels = ['poor', 'normal', 'good', 'best']
        flairlabels = ['__label__'+x for x in labels]
        data = self.df.copy()
        data = self.clean(data)
        bertdata = self.preprocess(data, [0, 1, 2, 3], data.shape[0], 512)
        fastdata = self.preprocess(data, labels, data.shape[0], 512)
        flairdata = self.preprocess(data, flairlabels, data.shape[0], 512, 1)
        self.savedata(bertdata, self.output_filename("bertdata/train.csv"), self.output_filename("bertdata/test.csv"))
        self.savedata(fastdata, self.output_filename("fastdata/train.csv"), self.output_filename("fastdata/test.csv"))
        self.savedata(flairdata, self.output_filename("flairdata/train.csv"), self.output_filename("flairdata/test.csv"), 1)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../../data/raw/data.csv')
    Dataset(filename).collect()
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()