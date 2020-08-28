import json

from flask import Flask, render_template, request

from preprocess import load_data
from fetch import fetch_tweet_and_replies
from search import search
from utils import read

app = Flask(__name__)
data = read('data/data.json')
rumours = list(data.keys())
threads = {k: v for rumour in data.values() for k, v in rumour.items()}
groups = read('data/groups.json')


def update_data():
    global data, rumours, threads, groups
    data = read('data/data.json')
    rumours = list(data.keys())
    threads = {k: v for rumour in data.values() for k, v in rumour.items()}
    groups = read('data/groups.json')


def replies(thread):
    # log(thread_id)
    # log(len(threads[thread_id].get('replies', dict()).values()))
    # return [log(tweet['text']) for tweet in threads[thread_id].get('replies', dict()).values()]
    # print('==')
    # print(len(thread.get('replies', dict()).values()))
    return [{
                'text': tweet['text'],
                'group': groups.get(tweet.get('id_str', 0), 'unknown')
            } for tweet in thread.get('replies', dict()).values()]



@app.route("/")
def index():
    return render_template('index.html', rumours=rumours)


@app.route("/fetch")
def fetch_tweet_page():
    return render_template('fetch.html')


@app.route("/fetch-tweet", methods=['POST'])
def fetch_tweet():
    tweet_id = request.form['tweet_id']
    reply_ids = request.form['replies_list']
    # print(tweet_id, reply_ids)
    fetch_tweet_and_replies(tweet_id, reply_ids)
    # print('loading')
    load_data()
    # print('updating')
    update_data()
    # print('rendering')
    return render_template('success.html')



@app.route("/style.css")
def style():
    return render_template('style.css')


@app.route('/text/<text>')
def search_text(text):
    print('searching for ', text.replace('%23', '#'))
    results = sorted(search(text).items(), key=lambda x: x[1], reverse=True)
    results = [{
                   "rating": rating,
                   "text": threads[id]['source']['text'],
                   "replies": replies(threads[id])
                   # [tweet['text'] for tweet in threads[id].get('replies', dict()).values()]
               } for id, rating in results]
    return json.dumps(results)


@app.route('/rumour/<rumour>')
def search_rumour(rumour):
    results = data[rumour].values()
    # print('-----')
    # print(len(results))
    # print(len(list(results)))
    # print(len(list(results)[0]))
    # print(list(results)[0]['source']['text'])
    # # print(json.dumps(list(results)[0]))
    # print('-----')
    results = [{
                   "text": thread['source']['text'],
                   "replies": replies(thread)
               } for thread in results]
    # print('dasdansuidnaskd')
    return json.dumps(results)


def main():
    global app, rumours
    print('Active rumours', rumours)
    app.run()


if __name__ == "__main__":
    main()