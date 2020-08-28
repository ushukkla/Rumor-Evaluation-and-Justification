import math
import re

from utils import read, write


def main():
    data = read('data/data.json')
    tweets = read('data/tweets.json')

    tf = {}
    idf = {}

    inverted_index = {}
    thread_count = sum(len(rumour) for rumour in data.values())

    for rumour_name, rumour in data.items():
        for thread_id, thread in rumour.items():
            tweets = [thread['source']] + list(thread.get('replies', dict()).values())
            word_index = {}
            word_count = 0

            for tweet in tweets:
                text = tweet['text']
                words = [word.lower() for word in re.findall("[\w#@']+", text)]

                for word in words:
                    word_index[word] = word_index.get(word, 0) + 1

                word_count += len(words)

            for word, count in word_index.items():
                tf[word] = tf.get(word, dict())
                tf[word][thread_id] = count / float(word_count)
                inverted_index[word] = inverted_index.get(word, dict())
                inverted_index[word][thread_id] = 1

    for word, thread_dict in inverted_index.items():
        idf[word] = math.log(thread_count / float(len(thread_dict)))

    write('data/tfidf.json', {"tf": tf, "idf": idf, "inverted_index": inverted_index})


if __name__ == '__main__':
    main()