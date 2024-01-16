import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op
from pyvi import ViTokenizer
from cytoolz import identity, concat, curry
import re
import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp
from data.batcher import tokenize
from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model_dir = 'rl_final'
beam_size = 1
diverse = 1.0
max_len = 30
cuda = False


with open(join(model_dir, 'meta.json')) as f:
    meta = json.loads(f.read())
if meta['net_args']['abstractor'] is None:
    # NOTE: if no abstractor is provided then
    #       the whole model would be extractive summarization
    assert beam_size == 1
    abstractor = identity
else:
    if beam_size == 1:
        abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
    else:
        abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
extractor = RLExtractor(model_dir, cuda=cuda)

def coll(batch):
        articles = list(filter(bool, batch))
        return articles

@app.route('/summrize', methods=['POST'])
def decode():
    start = time() 
    data = request.json
    paragraph = data['text']
    sentences = paragraph.split('.')
    trimmed_list = [ViTokenizer.tokenize(sentence.strip()) for sentence in sentences if sentence.strip()]


    raw_article_sentences = [trimmed_list]
    i = 0
    summary = []
    with torch.no_grad():
        ext_arts = []
        ext_inds = []
        tokenized_article_sentences = map(tokenize(None), raw_article_sentences)
        for raw_art_sents in tokenized_article_sentences:
            ext = extractor(raw_art_sents)[:-1] 
            if not ext:
                    # use top-5 if nothing is extracted
                    # in some rare cases rnn-ext does not extract at all
                ext = list(range(5))[:len(raw_art_sents)]
            else:
                ext = [i.item() for i in ext]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += [raw_art_sents[i] for i in ext]
        if beam_size > 1:
            all_beams = abstractor(ext_arts, beam_size, diverse)
            dec_outs = rerank_mp(all_beams, ext_inds)
        else:
            dec_outs = abstractor(ext_arts)
        for j, n in ext_inds:
            decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
            result = []

            for sentence in decoded_sents:

                words = sentence.split()

                new_sentence = []

                for word in words:

                    if '_' in word:

                        new_word = ' '.join(word.split('_'))

                        new_sentence.append(new_word)

                    else:

                        new_sentence.append(word)

                result.append(' '.join(new_sentence))
            concatenated_paragraph = '.'.join(result)   
            concatenated_paragraph = concatenated_paragraph.split('.')  
            concatenated_paragraph = [sentence.strip().capitalize() for sentence in concatenated_paragraph if sentence.strip()]
            concatenated_paragraph = [remove_duplicate_subphrases(sentence) for sentence in concatenated_paragraph]
            concatenated_paragraph = '. '.join(concatenated_paragraph)
            summary = concatenated_paragraph + "."
            i += 1

    return jsonify({'summary': summary})

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)


def remove_duplicate_subphrases(sentence):
    words = sentence.split()
    unique_words = []
    seen_subphrases = set()

    for word in words:
        # Use a regular expression to check if the word is a repeated subphrase of three words or more
        if len(word) >= 3 and re.match(r'\b(\w+)\b.*\b\1\b.*\b\1\b', word):
            continue

        unique_words.append(word)
        seen_subphrases.add(word)

    return ' '.join(unique_words)

def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)






if __name__ == '__main__':
  app.run(port=8080)

