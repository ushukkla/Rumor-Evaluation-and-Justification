import configparser
import os
import tweepy

from utils import write


def fetch_tweet_and_replies(tweet_id, reply_ids):
    # print('fetch_tweet_and_replies')
    create_folders(tweet_id)
    api = authenticate()

    tweet = api.get_status(tweet_id)
    persist_tweet(tweet, str(tweet.id), "source-tweet")

    structure = dict()
    structure[tweet_id] = list()

    for reply_id in reply_ids.split(','):
        reply = api.get_status(reply_id.strip())
        structure[tweet_id].append(reply_id)
        persist_tweet(reply, str(tweet.id), "replies")

    persist_structure(tweet_id, structure)


def create_folders(tweet_id):
    # print('create_folders')
    tweet_id_as_string = str(tweet_id)
    tweet_dir = "resources/dataset/rumoureval-data/random-rumours/" + tweet_id_as_string
    src_tweet_dir = tweet_dir + "/source-tweet"
    replies_tweet_dir = tweet_dir + "/replies"

    try:
        os.makedirs(tweet_dir)
        os.makedirs(src_tweet_dir)
        os.makedirs(replies_tweet_dir)
    except FileExistsError:
        return


def authenticate():
    # print('authenticate')
    config = configparser.ConfigParser()
    config.read('config/twitter.ini')

    consumer_key = config.get('Twitter', 'consumer_key')
    consumer_secret = config.get('Twitter', 'consumer_secret')
    access_key = config.get('Twitter', 'access_key')
    access_secret = config.get('Twitter', 'access_secret')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    return tweepy.API(auth)


def persist_tweet(tweet, parent_tweet_id, folder):
    # print('persist_tweet')
    tweet_id_as_string = str(tweet.id)
    tweet_dir = "resources/dataset/rumoureval-data/random-rumours/" + parent_tweet_id + "/" + folder
    try:
        write(tweet_dir + "/" + tweet_id_as_string + ".json", tweet._json)
    except FileExistsError:
        return


def persist_structure(tweet_id, structure):
    # print('persist_structure')
    tweet_id_as_string = str(tweet_id)
    tweet_dir = "resources/dataset/rumoureval-data/random-rumours/" + tweet_id_as_string
    try:
        write(tweet_dir + "/structure.json", structure)
    except FileExistsError:
        return
