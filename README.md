# Reddit Upvote Predictor

Reddit is a social news aggregation, web content rating, and discussion website. People registered on the website can submit content in links, text post, images, and as comments to other posts.

Similar to Facebook’s likes and Twitter’s  favorites, Reddit has an upvote system where users are allowed to vote up or down on others’ posts and comments.

The objective is explore reddit data and potentially design a model that predicts the number of upvotes a comment might receive.

The project started off as a term project for Natural Language Processing (CS 662). Group project with another student. Work was divided across the models; All work was initially pushed to Github classroom, moved to Github public for further development.

## Current Work:

Further feature extraction and analysis for potential improvements in results. Other ideas further feature extraction, such as adding user data (such as post karma score, comment karma score, past activity, account longevity) as features to predict comment score.

Alternatively, also working on predicting if a comment receives reddit currency (gold, silver etc). This can either be done as a binary classification (any currency or not) or multilabeled (Gold, silver, bronze, none).

Two different approaches are being worked on - one that only takes numerical features such user activity, karma score, longetivity into account, while the other adds in commment text as well.

# Running:

## Step 1: Clone from Github:

> git clone git@github.com:uabinf/nlp-fall-2019-project-reddit_upvote-predictor.git

## Step 2: Environment Setup:

> Run setup.sh to setup Virtual environment and required dependencies.

## Step 3: Setup Praw.ini (Required only for 4b)

Create a developer app on reddit, get the client_id, client_secret and replace them in praw.ini file. Also replace username and password fields with respective credentials.

## Step 4a:

Use existing data. Run SimpleTransformers and FastAI with their respective notebooks.

## Step 4b (Alternate):

Run script.py (or Datafetch.ipynb) to fetch comments from reddit. Run Dataprocess multiclass notebook for processing data for SimpleTransformers model and Dataprocess onehot notebook for processing data for FastAI model.

## Files:

**SimpleTransformersModel.ipynb** contains simpletransformers model.

**praw.ini** contains reddit setup details.

**FlairModel.ipynb** contains Flair model.

**fastmodel.ipynb** contains FastAI based model.



