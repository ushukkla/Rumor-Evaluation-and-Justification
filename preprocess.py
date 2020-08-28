import csv
import nltk

from os import path
from utils import read, write, folders, files
from nltk import TweetTokenizer

DATA_LOCATION = ".\\resources\\dataset\\rumoureval-data"
TRAIN = DATA_LOCATION + "\\..\\traindev\\rumoureval-subtaskA-train.json"
DEV = DATA_LOCATION + "\\..\\traindev\\rumoureval-subtaskA-dev.json"


def tweet(location):
    keys = ['text', 'id_str', 'created_at', 'in_reply_to_status_id_str', 'retweet_count', 'retweeted', 'entities', 'user']
    # user?
    data = read(location)
    return {key: data[key] for key in keys}


def load_data():
    # rumours = {k: {} for k in folders('.')}
    data = {}
    for rumour, r_location in folders(DATA_LOCATION):
        data[rumour] = {}
        for thread, t_location in folders(r_location):
            try:
                replies = files(path.join(t_location, 'replies'))
            except StopIteration:
                replies = []
            # print(rumour, thread, path.join(t_location, 'replies'))
            data[rumour][thread] = {
                "structure": read(path.join(t_location, 'structure.json')),
                "source": tweet(path.join(t_location, 'source-tweet', thread + '.json')),
                "replies": {id[:-5]: tweet(f) for id, f in replies}
            }

    write('data/data.json', data)
    return data


def walk(parent, node, result=None):
    if result is None:
        result = []
    for key, item in node.items():
        result.append({"from": parent, "to": key})
        if len(item):
            walk(key, item, result)
    return result


def create_table_json():
    data = read('data/data.json')
    all_tweets = {}
    for rumour, rumour_data in data.items():
        for x, thread in rumour_data.items():
            all_tweets[thread['source']['id_str']] = {
                'rumour': rumour,
                'text': thread['source']['text'],
                'id': thread['source']['id_str'],
            }
            if 'entities' in thread.items():
                all_tweets[key] = {
                    'media_url': thread['entities']
                    }
            if 'user' in thread.items():
                all_tweets[key] = {
                    'usr': thread['user']
                    }
            for key, tweet in thread['replies'].items():
                tokenized_tweet = tweet_tokenize(tweet['text'])
                all_tweets[key] = {
                    'rumour': rumour,
                    'text': tokenized_tweet,
                    'tags': tag_part_of_speech(tokenized_tweet),
                    'id': key,
                    'reply_to': tweet['in_reply_to_status_id_str']
                    # 'reply_to': all_tweets[tweet['in_reply_to_status_id_str']]['text']
                }

    for id, tweet in all_tweets.items():
        if 'reply_to' in tweet:
            tweet['reply_to'] = all_tweets[tweet['reply_to']]['text']

    for id, tweet in all_tweets.items():
        if 'usr' in tweet:
            tweet['usr'] = all_tweets[tweet['usr']]['verified']

    for id, tweet in all_tweets.items():
        if 'media_url' in tweet:
            tweet['media_url'] = all_tweets[tweet['media_url']]["media"]["media_url_https"]

    write('data/tweets.json', list(all_tweets.values()))
    to_csv(list(all_tweets.values()))
    print (all_tweets)
    return all_tweets.values()


def divide_train_dev(tweets):
    train_categories = read(TRAIN)
    dev_categories = read(DEV)
    train = []
    dev = []

    for tweet in tweets:
        if tweet.get('reply_to'):
            el = {
                'text': tweet['text'],
                'reply_to': tweet['reply_to']
            }

            if tweet['id'] in train_categories:
                el['group'] = train_categories[tweet['id']]
                train += [el]
                # train += [{
                #     'text': tweet['text'],
                #     'reply_to': tweet['reply_to'],
                #     'group': train_categories[tweet['id']]
                # }]
            else:
                el['group'] = dev_categories[tweet['id']]
                dev += [el]
                # dev += [{
                #     'text': tweet['text'],
                #     'reply_to': tweet['reply_to'],
                #     'group': dev_categories[tweet['id']]
                # }]
                # all += [el]

    write('data/train.json', train)
    write('data/dev.json', dev)
    write('data/groups.json', dict(train_categories.items() | dev_categories.items()))

#CSV file converter
def to_csv(data):
    keys = ['id', 'rumour', 'text', 'reply_to']

    with open('data/data.csv', 'w', encoding='utf-8') as file:
        csv_file = csv.writer(file)
        csv_file.writerow(keys)
        for item in data:
            if 'reply_to' in item:
                csv_file.writerow([item[key] for key in keys])
        for item in data:
            if 'media_http_url' in item:
                csv_file.writerow([item[key] for key in keys])
                print(media_http_url)
        for item in data:
            if 'usr' in item:
                csv_file.writerow([item[key] for key in keys])
        
#URL_summarizer
''''def getTextFromURL(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text

def summarizeURL(url, total_pars):
    url_text = getTextFromURL(url).replace(u"Â", u"").replace(u"â", u"")

    fs = FrequencySummarizer()
    final_summary = fs.summarize(url_text.replace("\n"," "), total_pars)
    return " ".join(final_summary)

url = raw_input("User_URL") #User_input_Variable
final_summary = summarizeURL(url, 5)
print final_summary


#Number to words
def Numbers_To_Words (number):
    dictionary = {'1': "one", '2': "two", '3': "three", '4': "four", '5': "five", '6': "six",
            '7': "seven", '8': "eight", '9': "nine", '0': "zero"}
    return " ".join(map(lambda x: dictionary[x], str(number)))


#spaces seprated
from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("words-by-frequency.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)

def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))


#abbreviation and phonetic
def translator(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "slang.txt"
        # File Access mode [Read Mode]
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    print(' '.join(user_string))
'''
def tweet_tokenize(text):
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    words = tokenizer.tokenize(text)
    return " ".join(words)


def tag_part_of_speech(text):
    tags = nltk.pos_tag(nltk.word_tokenize(text))
    result = []
    for (word, tag) in tags:
        result.append(tag)
    return " ".join(result)


def main():
    load_data()
    tweets = create_table_json()
    divide_train_dev(tweets)


if __name__ == "__main__":
    main()