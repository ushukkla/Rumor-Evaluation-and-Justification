import json
import os


def log(sth):
    print(sth)
    return sth


def read(file):
    with open(file) as data_file:
        return json.load(data_file)


def write(file, data):
    with open(file, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def folders(dirname):
    return ((f, os.path.join(dirname, f)) for f in next(os.walk(dirname))[1])


def files(dirname):
    return ((f, os.path.join(dirname, f)) for f in next(os.walk(dirname))[2])
