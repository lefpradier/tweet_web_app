import pickle as pkl
import spacy
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import re
import spacy_fastlang
import numpy as np


def preprocessing_transform(tweet):
    # load spacy model
    from_disk = pkl.load(open("backend/vectorizer.pkl", "rb"))
    vectorizer = TextVectorization.from_config(from_disk["config"])
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorizer.set_weights(from_disk["weights"])
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("language_detector")

    to_remove = set()

    # change stop words
    for word in nlp.Defaults.stop_words:
        if (
            "n't" in word
            or "no" in word
            or word.endswith("dn")
            or word.endswith("sn")
            or word.endswith("tn")
        ):
            to_remove.add(word)

    # custom select words you don't want to eliminate
    words_to_remove = [
        "for",
        "by",
        "with",
        "against",
        "shan",
        "don",
        "aren",
        "haven",
        "weren",
        "until",
        "ain",
        "but",
        "off",
        "out",
    ]
    for word in words_to_remove:
        if word in nlp.Defaults.stop_words:
            to_remove.add(word)
    nlp.Defaults.stop_words -= to_remove

    # Defining regex patterns.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = "@[^\s]+"
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    # lowercase the tweets
    tweet = tweet.lower().strip()

    # REMOVE all URls
    tweet = re.sub(urlPattern, "", tweet)

    # Remove @USERNAME
    tweet = re.sub(userPattern, "", tweet)

    # Replace all non alphabets.
    tweet = re.sub(alphaPattern, " ", tweet)

    # Spell check elision : Replace 3 or more consecutive letters by 2 letter.
    tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

    # apply nlp tokenizer model
    token_tweet = nlp(tweet)
    tweetwords = ""
    # keep only en tweet
    if (token_tweet._.language == "en") and (token_tweet._.language_score >= 0.5):
        for word in token_tweet:
            # Checking if the word is a stopword.
            if not word.is_stop:
                if len(word.text) > 1:
                    # Lemmatizing the word.
                    lem_word = word.lemma_
                    tweetwords += lem_word + " "
    tweetwords = vectorizer(np.array([tweetwords])).numpy()
    return tweetwords
