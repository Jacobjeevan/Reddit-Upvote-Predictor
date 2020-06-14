from pathlib import Path
import argparse, swifter, os, re, nltk
import pandas as pd
import numpy as np
from pycontractions import Contractions
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit

cont = Contractions(api_key="glove-twitter-100")
cont.load_models()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
stop_words.remove("not")
stop_words.remove("no")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class Dataset:

    def __init__(self, input_filepath, args):
        self.df = pd.read_csv(input_filepath)
        self.savepath = args.savepath
        self.charlimit = args.charlimit
        self.data = self.df.copy()[0:100]
        self.column = "comment_body"

    def spaceURLs(self):
        '''This method is used to enforce proper spacing
        Ex: In the data, you may have '[the image](https://image.xyz)';
        this method creates space between alt text ("the image") and the URL.'''
        self.data.loc[:, self.column] = self.data[self.column].str.replace('\[|\]', ' ', regex=True)

    def removeURL(self):
        self.data.loc[:, self.column] = self.data[self.column].str.replace('\(http\S+', 'URL', regex=True)

    def removeSymbols(self):
        self.data.loc[:, self.column] = self.data[self.column].str.replace('/r/', '', regex=True)
        self.data.loc[:, self.column] = self.data[self.column].str.replace('[^\.\'A-Za-z0-9]+', ' ', regex=True)

    def removeNumbers(self):
        self.data.loc[:, self.column] = self.data[self.column].str.replace('\S*\d\S*', '', regex=True)

    def processContractions(self):
        self.data["hasContractions"] = self.data[self.column].str.contains("'")
        self.data[self.column] = self.data.swifter.apply(lambda row: self.expandContractions(row["comment_body"]) if row["hasContractions"]
                                                    else row["comment_body"], axis=1)
        self.data.drop(["hasContractions"], axis=1, inplace=True)

    def expandContractions(self, text):
        return ''.join(list(cont.expand_texts([text], precise=True)))

    def processWords(self):
        self.data.loc[:, self.column] = self.data[self.column].str.replace("\.", ' ', regex=True)
        self.data.loc[:, self.column] = self.data[self.column].swifter.apply(
            lambda x: self.Lemmatize(x))

    def Lemmatize(self, text):
        tokens = word_tokenize(text)
        tokens = [x.lower() for x in tokens if x.lower() not in stop_words]
        #tokens = [stemmer.stem(x) for x in tokens]
        return ' '.join([lemmatizer.lemmatize(x, pos="v") for x in tokens])

    def removeCommentsOverLimit(self, charlimit):
        self.data = self.data.loc[self.data['comment_body'].apply(
            lambda x: len(x) <= charlimit), :]

    def createBins(self, labels):
        self.data["upvotes"] = pd.cut(
            self.data["upvotes"], bins=[-np.inf, 0., 3, 9, np.inf], labels=labels)

    def renameColumns(self):
        self.data.rename(columns={"comment_body": "text", "upvotes": "label"}, inplace=True)

    def prepare(self):
        self.spaceURLs()
        self.removeURL()
        self.removeNumbers()
        self.removeSymbols()
        self.processContractions()
        interim = self.makeFolder('../../data/interim/')
        self.data.to_csv(f"{interim}data_contractions_expanded.csv", index=False)
        self.processWords()
        self.data.to_csv(f"{interim}data_preprocessed.csv", index=False)
        self.data = self.data.dropna()
        labels = ['poor', 'normal', 'good', 'best']
        flairlabels = ['__label__'+x for x in labels]
        self.removeCommentsOverLimit(512)
        self.createBins(flairlabels)
        self.renameColumns()
        self.saveData(self.makeFolder('flairdata/'))

    def saveData(self, outputfolder):
        splits = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=42)
        for trainIndex, tempIndex in splits.split(self.data, self.data.label):
            pass
        trainSet = self.data.iloc[trainIndex, :]
        tempSet = self.data.iloc[tempIndex, :]
        splits = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=42)
        for testIndex, valIndex in splits.split(tempSet, tempSet.label):
            pass
        testSet = tempSet.iloc[testIndex, :]
        valSet = tempSet.iloc[valIndex, :]
        trainSet.to_csv(f"{outputfolder}/train.csv", index=False)
        testSet.to_csv(f"{outputfolder}/test.csv", index=False)
        valSet.to_csv(f"{outputfolder}/val.csv", index=False)

    def makeFolder(self, name=''):
        """Simple function to quickly fetch directory path and return the
        right filepath for saving files.
        Also creates the save folder if it doesn't already exist"""
        dirname = os.path.dirname(__file__)
        Path(os.path.join(dirname, self.savepath + name)
             ).mkdir(parents=True, exist_ok=True)
        return os.path.join(dirname, self.savepath + name)

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
    Dataset(filename, args).prepare()

if __name__ == '__main__':
    main()
