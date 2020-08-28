import shutil

from preprocess import main as preprocess
from index import main as create_index
from tfidf import main as tfidf
from classifier import main as create_classifier
from server import main as start_server


if __name__ == '__main__':
    shutil.rmtree('resources/dataset/rumoureval-data/random-rumours', True)
    print('running preprocess')
    preprocess()
    print('creating index')
    create_index()
    print('calculating tfidfs')
    tfidf()
    print('starting server')
    start_server()