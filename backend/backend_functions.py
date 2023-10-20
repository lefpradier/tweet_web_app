import pickle as pkl
import spacy
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import re
import spacy_fastlang
import numpy as np
from fastapi import HTTPException


def preprocessing_early_steps(tweet, vectorizer, raise_exceptions=True):
    """
    Premières étapes de preprocessing des tweets
    Paramètres :
    - tweet : string
    - vectorizer : objet vectorizer Keras
    - raise_exceptions : Boolean, détermine si les exceptions doivent être rapportées
    """
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

    # todo : empty tweet
    if raise_exceptions and len(tweet.strip()) == 0:
        raise HTTPException(
            status_code=400, detail="Tweet is empty or contains only hypertext links"
        )
    # apply nlp tokenizer model
    token_tweet = nlp(tweet)
    tweetwords = ""
    # keep only en tweet
    cvocab = 0
    cnvocab = 0

    for word in token_tweet:
        # Checking if the word is a stopword.
        if not word.is_stop:
            if len(word.text) > 1:
                # todo : check vocab and send warning
                # Lemmatizing the word.
                lem_word = word.lemma_
                if lem_word in vectorizer.get_vocabulary():
                    tweetwords += lem_word + " "
                    cvocab += 1
                else:
                    cnvocab += 1
    return (
        tweetwords,
        cvocab,
        cnvocab,
        token_tweet._.language,
        token_tweet._.language_score,
    )


def preprocessing_transform(
    tweet, vectorizer_path="backend/vectorizer.pkl", raise_exceptions=True
):
    """
    Préprocessing complet d'un tweet
    Paramètres :
    - tweet : string
    - vectorizer_path : string, chemin vers le vectorizer Keras
    - raise_exceptions : Boolean, détermine si les exceptions doivent être rapportées
    """
    # load spacy model
    from_disk = pkl.load(open(vectorizer_path, "rb"))
    vectorizer = TextVectorization.from_config(from_disk["config"])
    vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorizer.set_weights(from_disk["weights"])

    tweetwords, cvocab, cnvocab, language, l_score = preprocessing_early_steps(
        tweet, vectorizer, raise_exceptions
    )

    tweetwords = vectorizer(np.array([tweetwords])).numpy()
    return (tweetwords, cvocab, cnvocab, language, l_score)
