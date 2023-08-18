#!/usr/bin/env python3
"""Evaluation Metrics"""
import numpy as np


def transform_grams(references, sentence, n):
    """"Transform ngrams"""
    if n == 1:
        return references, sentence
    else:
        ngram_sent = []
        sent_length = len(sentence)
        for i, word in enumerate(sentence):
            count = 0
            w = word
            for j in range(1, n):
                if sent_length > i + j:
                    w += ' ' + sentence[i + j]
                    count += 1
            if count == n - 1:
                ngram_sent.append(w)
        ngram_ref = []
        for ref in references:
            n_ref = []
            ref_len = len(ref)
            for i, word in enumerate(ref):
                count = 0
                w = word
                for j in range(1, n):
                    if ref_len > i + j:
                        w += ' ' + ref[i + j]
                        count += 1
                if count == n - 1:
                    n_ref.append(w)
            ngram_ref.append(n_ref)
        return ngram_ref, ngram_sent


def ngram_bleu(references, sentence, n):
    """Ngram bleu"""
    ngram_ref, ngram_sent = transform_grams(references, sentence, n)
    ngram_sent_len = len(ngram_sent)
    sentence_length = len(sentence)
    sentenc_dict = {word: ngram_sent.count(word) for word in ngram_sent}
    ref_dict = {}
    for ref in ngram_ref:
        for gram in ref:
            if ref_dict.get(gram) is None or ref_dict[gram] < ref.count(gram):
                ref_dict[gram] = ref.count(gram)
    matchings = {word: 0 for word in ngram_sent}
    for ref in ngram_ref:
        for gram in matchings.keys():
            if gram in ref:
                matchings[gram] = max(matchings[gram], sentenc_dict[gram])
    for gram in matchings.keys():
        if ref_dict.get(gram) is not None:
            matchings[gram] = min(ref_dict[gram], matchings[gram])
    precision = sum(matchings.values()) / ngram_sent_len
    index = np.argmin([abs(len(i) - sentence_length) for i in references])
    references_len = len(references[index])
    if sentence_length > references_len:
        BLEU = 1
    else:
        BLEU = np.exp(1 - float(references_len) / sentence_length)
    BLEU_Score = BLEU * precision
    return BLEU_Score
