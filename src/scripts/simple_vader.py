#!/usr/bin/env python3
"""
Simple VADER

This module provides classes for sentiment analysis of text, particularly suited for
short, informal texts like social media posts.

This is a very simple representation of the VADER (Valence Aware Dictionary and sEntiment Reasoner)

Classes:
    SentiText: Preprocesses input text by tokenizing and cleaning words and emoticons
        for sentiment analysis.

    SentimentIntensityAnalyzer: Analyzes text to compute sentiment intensity scores,
        including positive, negative, neutral, and compound scores.
"""
# TODO: add capitalization boosters (currently just checks for cap differential, but doesn't apply C_INCR)
# TODO: apply booster words properly (e.g., scalar_inc_dec should consider distance, up to 3 words before a sentiment word)
# TODO: handle negation words (e.g., "not good" should invert valence)
# TODO: implement contrastive conjunction handling (e.g., shift weight after "but")
# TODO: implement sentiment shift based on degree modifiers like "least" before a word
# TODO: consider idioms and special case phrases (e.g., "the bomb", "bad ass", "yeah right")
# TODO: handle emoticons and emojis (e.g., :) or 😢 contributing to valence)
# TODO: account for repeated letters as emphasis (e.g., "sooo good")
# TODO: add punctuation emphasis (already partially implemented with _punctuation_emphasis, but not fully integrated)
# TODO: handle increment/decrement scope (right now, scalar_inc_dec only checks one word without backtracking or accumulation)
# TODO: add handling for "never", "without", and other complex negation scopes
# TODO: implement heuristic tuning (e.g., optimizing weights like C_INCR and punctuation amplification empirically)
# TODO: optimize performance and structure (e.g., reduce repeated code or unnecessary recomputations)

import os
import math
import string
from inspect import getsourcefile
from typing import Dict, List


# BOOSTER Increments and Decrements
B_INCR = 0.293
B_DECR = -0.293

# rating increase for using ALLCAPs to emphasize word
C_INCR = 0.733
N_SCALAR = -0.74



BOOSTER_DICT = \
        {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR,
         "completely": B_INCR, "considerable": B_INCR, "considerably": B_INCR,
         "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormous": B_INCR, "enormously": B_INCR,
         "entirely": B_INCR, "especially": B_INCR, "exceptional": B_INCR, "exceptionally": B_INCR,
         "extreme": B_INCR, "extremely": B_INCR,
         "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR, "frackin": B_INCR, "fracking": B_INCR,
         "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR,
         "fuckin": B_INCR, "fucking": B_INCR, "fuggin": B_INCR, "fugging": B_INCR,
         "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR,
         "incredible": B_INCR, "incredibly": B_INCR, "intensely": B_INCR,
         "major": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
         "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
         "so": B_INCR, "substantially": B_INCR,
         "thoroughly": B_INCR, "total": B_INCR, "totally": B_INCR, "tremendous": B_INCR, "tremendously": B_INCR,
         "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utter": B_INCR, "utterly": B_INCR,
         "very": B_INCR,
         "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
         "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
         "less": B_DECR, "little": B_DECR, "marginal": B_DECR, "marginally": B_DECR,
         "occasional": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
         "scarce": B_DECR, "scarcely": B_DECR, "slight": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
         "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}


def allcap_differencial(words: List[str]) -> bool:
    """
    Check whether just some words in the input are ALL CAPS
    :param list words: The words to inspect
    :returns: `True` if some but not all items in `words` are ALL CAPS
    """
    is_different = False
    allcap_words = 0

    for word in words:
        if word.isupper():
            allcap_words += 1

    cap_differential = len(words) - allcap_words
    if 0 < cap_differential < len(words):
        is_different = True

    return is_different


def normalize(score, alpha=15) -> float:
    """
    Normalize the score to be between -1 and 1 using an alpha that
    approximates the max expected value
    """
    norm_score = score / math.sqrt((score * score) + alpha)
    if norm_score < -1.0:
        return -1.0
    elif norm_score > 1.0:
        return 1.0
    else:
        return norm_score


def scalar_inc_dec(word, valence, is_cap_diff) -> float:
    """
    Check if the preceding words increase, decrease, or negate/nullify the
    valence
    """
    scalar = 0.0
    word_lower = word.lower()

    if word_lower in BOOSTER_DICT:
        scalar = BOOSTER_DICT[word_lower]

        if valence < 0:
            scalar *= -1
    return scalar


class SentiText(object):
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text):
        if not isinstance(text, str):
            text = str(text).encode('utf-8')

        self.text = text
        self.words_and_emoticons = self._words_and_emoticons()
        self.is_cap_diff = allcap_differencial(self.words_and_emoticons)

    @staticmethod
    def _strip_punc_if_word(token: str) -> str:
        """
        Removes all trailing and leading punctuation
        If the resulting string has two or fewer characters,
        then it was likely an emoticon, so return original string
        (ie ":)" stripped would be "", so just return ":)"
        """
        stripped = token.strip(string.punctuation)

        if len(stripped) <= 2:
            return token
        return stripped 


    def _words_and_emoticons(self) -> List[str]:
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        raw_tokens = self.text.split()
        stripped = list(map(self._strip_punc_if_word, raw_tokens))
        return stripped


class SentiementIntensityAnalyzer:
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(self, lexicon_file="vader_lexicon.txt"):
        _this_module_file_path_ = os.path.abspath(getsourcefile(lambda: 0))
        lexicon_full_filepath = os.path.join(os.path.dirname(_this_module_file_path_), lexicon_file)

        with open(lexicon_full_filepath, 'r') as file:
            self.lexicon_full_filepath = file.read()

        self.lexicon = self.make_lex_dict()


    def make_lex_dict(self) -> Dict[str, float]:
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_full_filepath.rstrip('\n').split('\n'):
            if not line:
                continue
            (word, measure) = line.strip().split('\t')[0:2]
            lex_dict[word] = float(measure)
        return lex_dict


    def polarity_scores(self, text: str) -> Dict[str, float]:
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        sentitext = SentiText(text)

        sentiments: List[float] = []
        words_and_emoticons = sentitext.words_and_emoticons

        for i, item in enumerate(words_and_emoticons):
            valence = 0

            if item.lower() in BOOSTER_DICT:
                sentiments.append(valence)
                continue

            # get the sentiment scores of the words
            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        # calculate the {"positive": value, "negative": value, "neutral": value, "compound": value}
        valence_dict = self.score_valence(sentiments, text)

        return valence_dict


    def sentiment_valence(self, valence: float, sentitext: SentiText, item: str, i: int, sentiments: List[float]):
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()

        if item_lowercase in self.lexicon:
            valence = self.lexicon[item_lowercase]

            # check for "no" as negation for adjacent lexicon vs "no" as its own stand-alone lexicon item
            if item_lowercase == "no" and i != len(words_and_emoticons) - 1 and words_and_emoticons[i + 1].lower() in self.lexicon:
                # don't use valence of "no" as a lexicon item. Instead set it's valence to 0.0 and negate the next item
                valence = 0.0

            if (i > 0) and words_and_emoticons[i - 1].lower() == "no" \
                    or (i > 1 and words_and_emoticons[i - 2] == "no" ) \
                    or (i > 2 and words_and_emoticons[i - 3] == "no" and words_and_emoticons[i - 1].lower() in ["or", "nor"]):
                        valence = self.lexicon[item_lowercase] * N_SCALAR

        sentiments.append(valence)
        return sentiments


    def _punctuation_emphasis(self, text) -> float:
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        return ep_amplifier + qm_amplifier


    @staticmethod
    def _amplify_ep(text: str):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if (ep_count) > 4:
            ep_count = 4

        em_amplifier = ep_count * 0.292
        return em_amplifier


    @staticmethod
    def _amplify_qm(text: str) -> float:
        # check from added emphasis from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0.0

        if qm_count > 1:
            if qm_count <= 3:
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier


    @staticmethod
    def _sift_sentiment_scores(sentiments: List[float]):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (float(sentiment_score) + 1)  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (float(sentiment_score) - 1)  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count


    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))

            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = normalize(sum_s)

            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = \
                {"neg": round(neg, 3),
                 "neu": round(neu, 3),
                 "pos": round(pos, 3),
                 "compound": round(compound, 4)}

        return sentiment_dict


if __name__ == "__main__":
    # EXAMPLE SENTENCES TO SEE THE SIMPLE VADER IN ACTION 
    sia = SentiementIntensityAnalyzer()

    positive = "I absolutely love this new coffee shop; the ambiance is fantastic."
    pos_neg, pos_neu, pos_pos, pos_compound = sia.polarity_scores(positive).values()

    print("------------------------------------------------------------------------------------------------------------------------------------\n")
    print(positive)
    print(f"With SIMPLE VADER, the sentiment score for the text came about like --> Negativity: {pos_neg}, Neutrality: {pos_neu}, Positivity: {pos_pos}, Compound Value: {pos_compound}")
    print("------------------------------------------------------------------------------------------------------------------------------------\n")

    negative = "The customer service was terribly slow and unhelpful today."
    neg_neg, neg_neu, neg_pos, neg_compound = sia.polarity_scores(negative).values()

    print("------------------------------------------------------------------------------------------------------------------------------------\n")
    print(negative)
    print(f"With SIMPLE VADER, the sentiment score for the text came about like --> Negativity: {neg_neg}, Neutrality: {neg_neu}, Positivity: {neg_pos}, Compound Value: {neg_compound}")
    print("------------------------------------------------------------------------------------------------------------------------------------\n")

    neutral = "The cat sat on the mat."
    neu_neg, neu_neu, neu_pos, neu_compound = sia.polarity_scores(neutral).values()

    print("------------------------------------------------------------------------------------------------------------------------------------\n")
    print(neutral)
    print(f"With SIMPLE VADER, the sentiment score for the text came about like --> Negativity: {neu_neg}, Neutrality: {neu_neu}, Positivity: {neu_pos}, Compound Value: {neu_compound}")
    print("------------------------------------------------------------------------------------------------------------------------------------\n")

