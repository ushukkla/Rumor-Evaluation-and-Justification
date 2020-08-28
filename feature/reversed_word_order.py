REVERSED_ORDER_TAGS = ["NN PRP", "NNP PRP VBP", "WP VB", "WRB VB PRP", "WDT NNS", "VB DT"]


def reversed_word_order(tags):
    return [x in tags for x in REVERSED_ORDER_TAGS]
