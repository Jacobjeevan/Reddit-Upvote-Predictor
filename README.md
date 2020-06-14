# Reddit Upvote Predictor

Reddit is a social news aggregation, web content rating, and discussion website. People registered on the website can submit content in links, text post, images, and as comments to other posts.

Similar to Facebook’s likes and Twitter’s  favorites, Reddit has an upvote system where users are allowed to vote up or down on others’ posts and comments.

__So why reddit data?__

We chose reddit primarily because we wanted to explore our favorite movie subreddits. Initially we were thinking of doing sentiment analysis (and looking at change in sentiment over time, for instance: Ben Affleck's Batman casting), but had to reduce the scope due to the arduous task of manually labeling the data.

So we pivoted, and decided to design a model that would predict the number of upvotes a comment might receive.

Notes:

The project started off as a term project for Natural Language Processing (CS 662); It was an open ended group project with another student. Work was divided across the models; My partner worked on SimpleTransformers model (currently not in this repo due to changes in the library), while I worked on FlairNLP and FastAI (currently removed from repo) based models; All work was initially pushed to Github classroom, moved to Github public for further development.

## Current/Future Work:

Popular NLP based models were used because of its relevance within the class. For future work, I would like to go back to basics:

- Rework any existing code, fix bugs and repackage as python code
- Get word vectors (Sklearn's Vectorizer methods) and use them to train simpler and more interpretable models such as Logistic Regression and SVM.
- Repeat the same with word embeddings (Glove/Flair Embeddings)

# Running:

## Step 1: Clone from Github:

> git clone https://github.com/Jacobjeevan/Reddit-Upvote-Predictor

## Step 2: Environment Setup:

> Run setup.sh to setup Virtual environment and required dependencies.

## Step 3: Setup Praw.ini

> Create a developer app on reddit, get the client_id, client_secret and replace them in praw.ini file. Also replace username and password fields with respective credentials.

## Step 4: Run fetch_dataset.py

> Run fetch_dataset.py to scrape and save relevant information from user input subreddit.

## Step 5: Run preprocess_dataset.py

> Run preprocess_dataset.py to preprocess and clean the scraped data for training the models.

## Step 6: Run notebooks

> Run the notebooks under src/reports folder to run the models. Current work also includes converting the models into scripts for future use.


