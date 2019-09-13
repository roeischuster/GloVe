#!/usr/bin/python
from embedding_models.glove_model import GloveSteps, execute_glove_step_internal, glove_model_to_word2vec
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import random
import resources as r
from scipy import stats
import sys
import os
import logging
import datetime
import logging
from os import path
from os.path import join
from argparse import ArgumentParser
import numpy as np
import pickle
import scipy
import words_info as wi
import resources as r
import itertools as it
import conf
from conf import *
from corpora.text_dump import TextDump
from collections import defaultdict
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix
import hdrtypes as h
import struct
from gensim.models import Word2Vec


def opt_parse():
     parser = ArgumentParser()
     parser.add_argument("-ct", "--corpus_type", dest="corpus_type", help="corpus type", metavar="CORPUS_TYPE")
     parser.add_argument("-n", "--native", dest="native", help="use glove's native cooccurrence count (fitted to count w2v)", action='store_true', default=False)
     options = parser.parse_args()
     #if not options.filename:
     #	parser.error("No input file")
     return options

sample = 1e-3
window = 5
word2vec_params = {
        'min_count': 5,
        'negative': 5,
        'sample': 0,
        'size': 100,
        'window': window,
        'workers': 8,
        'sg': 1,
	'sample':sample
    }

native_params = {
        'VERBOSE': '2',
        'MEMORY': '4.0',
        'VOCAB_MIN_COUNT': '5',
        'MAX_VOCAB': '100000000',
        'VECTOR_SIZE': '100',
        'WINDOW_SIZE': window,
        'NUM_THREADS': '12',
}


def count_cooc_glovevocab(sentences, vocab):
	coocs = defaultdict(float)
	n = len(vocab)

	#calculating proabilities, see ~/.local/lib/python3.6/site-packages/gensim/models/word2vec.py lines 1721-1732
	#TODO calc threshold_count
	#threshold_count = sample*??
	for word, vals in vocab.items():
		count = vals['count']
		word_probability = min((sqrt(count / threshold_count) + 1) * (threshold_count / count), 1.0)
		vals['sample_int'] = int(round(word_probability * 2**32))

	coocs = lil_matrix((n, n))
	for sentence in sentences:
		word_vocabs = [vocab[w] for w in sentence if w in vocab.keys()
					   and vocab[w]['prob'] > np.random.rand() * 2 ** 32]
		for pos, word in enumerate(word_vocabs):
			#reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
	
			# now go over all words from the (reduced) window, predicting each one in turn
			#start = max(0, pos - model.window + reduced_window)
			#for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
			start = max(0, pos - window)
			for pos2, word2 in enumerate(word_vocabs[start:(pos + window + 1)], start):
				# don't train on the `word` itself
				if pos2 != pos:
					dist = np.abs(pos2-pos)-1#subtract 1 since we sample from [0, windowsize-1], not windowsize
					dist_weight = 1-dist/window
					coocs[word.index, word2.index] += dist_weight
	
		#result += len(word_vocabs)
	return coocs



def count_cooc(sentences, model, glovevocab):
	coocs = defaultdict(float)
	n = len(glovevocab)
	coocs = lil_matrix((n, n))
	for sentence in sentences:
		word_vocabs = [(w, model.wv.vocab[w]) for w in sentence if w in model.wv.vocab
					   and model.wv.vocab[w].sample_int > model.random.rand() * 2 ** 32]
		for pos, (w, word) in enumerate(word_vocabs):
			#reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
	
			# now go over all words from the (reduced) window, predicting each one in turn
			#start = max(0, pos - model.window + reduced_window)
			#for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
			start = max(0, pos - model.window)
			for pos2, (w2, word2) in enumerate(word_vocabs[start:(pos + model.window + 1)], start):
				# don't train on the `word` itself
				if pos2 != pos:
					dist = np.abs(pos2-pos)-1#subtract 1 since we sample from [0, windowsize-1], not windowsize
					dist_weight = 1-dist/model.window
					coocs[glovevocab[w]['index'], glovevocab[w2]['index']] += dist_weight
	
		#result += len(word_vocabs)
	return coocs


if __name__ == "__main__":
	options = opt_parse()
	cache_dir = h.cttoglpath(options.corpus_type)

	textdump = TextDump(options.corpus_type)
	if options.native:
		override_params = {
			'vocab_path' : os.path.join(cache_dir, 'glove', 'vocab_w2v.dat'),
			'CO_OCCURRENCE_FILE': os.path.join(cache_dir, 'glove', 'co_occurrences_w2v_native.dat'),
        		'BUILD_DIR': os.path.join(os.path.abspath(os.path.curdir), "w2v_cooccurrences", "build")  # Should Be global
			}
		assert(os.path.exists(override_params['BUILD_DIR']))
		for glove_step in [GloveSteps.BUILD_VOCAB,
	                   GloveSteps.BUILD_CO_OCCURRENCE,
	                   ]:
	    		execute_glove_step_internal(None, None, glove_step, cache_dir, override_params, **native_params)
		quit()

	w2v = Word2Vec(tqdm(textdump), iter=0, **word2vec_params)
	vocab = r.get_vocab(cache_dir)
	coocs = count_cooc(tqdm(textdump.get_texts()), w2v, vocab)
	coocs_coord = coocs.tocoo()
	coocs_len = coocs.getnnz()
	with open(os.path.join(cache_dir, 'glove', 'co_occurrences_w2v.dat'), 'wb') as coocout:
		prev_i = 0
		for i,j,c in tqdm(zip(coocs_coord.row, coocs_coord.col, coocs_coord.data), total=coocs_len):
			assert(i>=prev_i)
			coocout.write(struct.pack('iid', i, j, c))
			prev_i = i

