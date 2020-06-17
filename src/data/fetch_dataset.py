#!/usr/bin/env python
# coding: utf-8
import praw
import sys
import pandas as pd
import time
import argparse
# Get credentials from DEFAULT instance in praw.ini
reddit = praw.Reddit()


class fetch_dataset:

    # Checkpoint defines savepoints (save everytime n number of comments are collected).
    def __init__(self, sub_reddit, checkpoint=None, minimum=None):
        self.exectime = time.time()
        self.ids = []
        self.checkpoint = checkpoint
        self.interval = checkpoint
        self.minimum = minimum
        if self.minimum < self.checkpoint:
            print(
                "The minimum number (default: 200k) of records has to larger than checkpoint (default: 10k)")
            sys.exit()
        self.commentdata = {"comment_body": [],
                            "upvotes": []}
        self.reddit = praw.Reddit()
        self.subreddit = self.reddit.subreddit(sub_reddit).top(limit=None)

    def addTo(self):
        for submission in self.subreddit:
            if (submission.id not in self.ids):
                self.ids.append(submission.id)
                submission.comments.replace_more(limit=None)
                all_comments = submission.comments.list()
                for comment in all_comments:
                    self.retrieveComment(comment)
                self.saveOrExitConditions()

    def retrieveComment(self, comment):
        if (not self.skipCommentConditions):
            self.commentdata["comment_body"].append(comment.body)
            self.commentdata["upvotes"].append(comment.score)

    def skipCommentConditions(self, comment):
        if (comment.author is None or comment.body is None or self.checkIfSuspended(comment) != None):
            return True
        return False

    def checkIfSuspended(self, comment):
        suspended = None
        try:
            suspended = comment.author.is_suspended
        except praw.exceptions.RedditAPIException:
            pass
        return suspended

    def saveOrExitConditions(self):
        length = len(self.commentdata["upvotes"])
        if (length < self.checkpoint):
            pass
        else:
            self.saveFiles(length)
            self.exitConditions()

    def saveFiles(self, length):
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("Collected {} records so far; Saving in progress. Time now: {}".format(
            length, current_time))
        data_f = pd.DataFrame(self.commentdata)
        data_f.to_csv('../../data/raw/comment_data.csv',
                        mode="w", index=False)
        self.checkpoint += self.interval
        
    def exitConditions(self):
        length = len(self.commentdata["upvotes"])
        if (length > self.minimum):
            self.exectime = ((time.time() - self.exectime) / (60*60))
            print("Collected {} records so far; Exiting now. Total execution time: {} hours".format(
                length, self.exectime))
            sys.exit()

def build_parser():
    checkpoint = 10000
    minimum = 200000
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sub_reddit", help="Specify the subreddit to scrape from")
    parser.add_argument("-m", "--minimum", help="Specify the minimum number of data records to collect",
                        type=int, default=checkpoint)
    parser.add_argument("-c", "--checkpoint",
                        help="Save the file every c comments", type=int, default=minimum)
    return parser



def main():
    parser = build_parser()
    args = parser.parse_args()
    if (args.minimum and args.checkpoint):
        Scraper(args.sub_reddit, checkpoint=args.checkpoint,
                minimum=args.minimum).addTo()
    if (args.minimum):
        Scraper(args.sub_reddit, minimum=args.minimum).addTo()
    if (args.checkpoint):
        Scraper(args.sub_reddit, checkpoint=args.checkpoint).addTo()
    Scraper(args.sub_reddit).addTo()


if __name__ == "__main__":
    main()
