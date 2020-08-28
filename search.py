import re

from utils import read, write

tfidf = read('data/tfidf.json')
tf, idf, inverted_index = tfidf['tf'], tfidf['idf'], tfidf['inverted_index']


def search(text):
    words = [word.lower() for word in re.findall("[\w#@']+", text)]
    words = [word for word in words if idf.get(word, 0) > 1]
    all_documents = {}
    tfidfs = {}
    for word in words:
        documents = inverted_index.get(word, {});
        all_documents = dict(all_documents.items() | documents.items())

        for doc in documents.keys():
            tfidfs[doc] = tfidfs.get(doc, 0) + idf[word] * tf.get(word, dict()).get(doc)

    write('data/results.json', tfidfs)

    return tfidfs


if __name__ == '__main__':
    search('shots fired')
