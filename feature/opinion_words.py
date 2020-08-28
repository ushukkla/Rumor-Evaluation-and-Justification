def opinion_words_count(text, lexicon):
    split = text.split()
    total = len(split)
    if total == 0:
        return 0
    count = 0
    for word in split:
        if word in lexicon:
            count += 1
    return count
