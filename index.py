from utils import read, write


def count_occurrences(tweets):
    words = {}
    bigrams = {}
    for tweet in tweets:
        text = tweet['text'].split(' ')

        last_word = None
        for word in text:
            words[word] = words.get(word, 0) + 1

            if last_word:
                bigram = last_word + ' ' + word
                bigrams[bigram] = bigrams.get(bigram, 0) + 1

            last_word = word
    return words, bigrams


def create_index(words, bigrams):
    word_index = {}
    bigram_index = {}

    i = 0
    for word, count in words.items():
        if count > 1:
            word_index[word] = i
            i += 1

    i = 0
    for bigram, count in bigrams.items():
        if count > 1:
            bigram_index[bigram] = i
            i += 1

    # words = {k: v for k, v in words.items() if word_index.get(k)}
    # bigrams = {k: v for k, v in bigrams.items() if bigram_index.get(k)}

    # write('data/count.json', {"words": words, "bigrams": bigrams})

    return {"words": word_index, "bigrams": bigram_index}


def print_common_occurrences_counts(words, bigrams):
    print(len(words))
    for i in list(range(1, 10)) + [20, 40, 80, 160, 320, 640, 1280]:
        common_words = 0
        for word, count in words.items():
            if count > i:
                common_words += 1
        print(i, common_words)

    print(len(bigrams))
    for i in list(range(1, 10)) + [20, 40, 80, 160, 320, 640, 1280]:
        common_words = 0
        for word, count in bigrams.items():
            if count > i:
                common_words += 1
        print(i, common_words)

    return


def main():
    train = read('data/train.json')
    tweets, bigrams = count_occurrences(train)
    # print_common_occurrences_counts(tweets, bigrams)
    index = create_index(tweets, bigrams)
    write('data/index.json', index)


if __name__ == '__main__':
    main()
