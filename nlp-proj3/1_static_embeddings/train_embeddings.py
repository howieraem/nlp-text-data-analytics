from concurrent.futures import ProcessPoolExecutor
from gensim.models import Word2Vec
import math
import nltk
import numpy as np
import os
import random
from scipy.sparse import csr_matrix, lil_matrix, load_npz, save_npz
from scipy.sparse.linalg import svds
from time import sleep
from typing import *


random.seed(0)
np.random.seed(0)
DEFAULT_BROWN_PATH = 'data/brown.txt'   # change this to your local path to brown.txt
N_WORKERS = 8


def load_brown(path_to_brown: str) -> Tuple[List, Dict, List]:
    """Extracts vocabulary, word-to-index mapping and sentences from the Brown corpus."""
    raw_vocab = set()
    sentences = []
    for line in open(path_to_brown, 'r').readlines():
        line = line.strip().lower().replace('``', '').replace(',', '').replace('\'\'', '').replace('.', '')
        tokens = nltk.word_tokenize(line)
        sentences.append(tokens)  # assuming each line is a sentence
        raw_vocab.update(tokens)
    vocab = sorted(raw_vocab)
    word2id = dict(zip(vocab, range(len(vocab))))
    return vocab, word2id, sentences


def build_brown_coo(word2id: Dict, sentences: List[List[str]], window_sz: int):
    """Constructs vocabulary and co-occurrence matrix from the Brown corpus."""
    serialize_filename = 'svd-w%d.npz' % window_sz
    try:
        return load_npz(serialize_filename)
    except:
        n = len(word2id)
        raw_mat = lil_matrix((n, n), dtype=np.int32)
        for sentence in sentences:
            length = len(sentence)
            for i, word in enumerate(sentence):
                wid = word2id[word]
                for j in range(max(i - window_sz, 0), min(i + window_sz, length)):
                    if j == i:
                        continue
                    raw_mat[wid, word2id[sentence[j]]] += 1
        csr_m = raw_mat.tocsr()
        save_npz(serialize_filename, csr_m)
        return csr_m


def log2(n):
    return math.log(n, 2)


def ppmi_ij(f_ij: float, s_j_f_i: float, s_i_f_j: float, s_f: float) -> float:
    """
    PMI_ij = log2(p_ij / ((p_i*) * p(_*j)))
           = log2((f_ij / s) / ((s_j[i] / s) * (s_i[j] / s)))
           = log2(f_ij / ((s_j[i] * s_i[j]) / s))
           = log2(f_ij * s / (s_j[i] * s_i[j]))
           = log2(f_ij) + log2(s) - log2(s_j[i]) - log2(s_i[j])
    """
    return max(log2(f_ij) + log2(s_f) - log2(s_j_f_i) - log2(s_i_f_j), 0)


def ppmi(coo_mat: csr_matrix) -> csr_matrix:
    """Builds the PPMI matrix from the co-occurrence matrix."""
    s_f = coo_mat.sum()
    s_i_f = coo_mat.sum(axis=0)
    s_j_f = coo_mat.sum(axis=1)
    ppmi_mat = lil_matrix(coo_mat.shape, dtype=float)
    for i, j in zip(*coo_mat.nonzero()):
        ppmi_value = ppmi_ij(coo_mat[i, j], s_j_f[i, 0], s_i_f[0, j], s_f)
        if ppmi_value > 0:
            ppmi_mat[i, j] = ppmi_value
    return ppmi_mat.tocsr()


def train_svd(cm: csr_matrix,
              vocab: List[str],
              window_size: int,
              vec_dimension: int):
    """Produces a SVD-based model given context window size and dimension k."""
    pm = ppmi(cm)
    u, s, vt = svds(pm, k=vec_dimension)

    # Since `s` here is a vector rather than a diagonal matrix,
    # should operate with `*` instead of `@`.
    w = u * np.sqrt(s)

    model_filename = 'svd-w%d-k%d.txt' % (window_size, vec_dimension)
    with open(model_filename, 'w+') as f:
        for i, row in enumerate(w):
            if not np.any(row):
                # ignore row with all zeros
                continue
            values = ' '.join([str(v) for v in row])
            f.write('%s %s\n' % (vocab[i], values))


def train_all_svd(path_to_brown: str = DEFAULT_BROWN_PATH):
    # Removes previous attempts
    for fname in os.listdir('.'):
        if fname.startswith('svd-'):
            os.remove(fname)

    # Loads corpus
    vocab, word2id, sentences = load_brown(path_to_brown)

    # Generates co-occurrence matrix in parallel
    with ProcessPoolExecutor(3) as exe:
        f10 = exe.submit(build_brown_coo, word2id, sentences, 10)
        f5 = exe.submit(build_brown_coo, word2id, sentences, 5)
        f2 = exe.submit(build_brown_coo, word2id, sentences, 2)
        cm2 = f2.result()
        cm5 = f5.result()
        cm10 = f10.result()
    sleep(1)

    # Generates SVD models in parallel
    with ProcessPoolExecutor(N_WORKERS) as exe:
        exe.submit(train_svd, cm10, vocab, 10, 300)
        exe.submit(train_svd, cm10, vocab, 10, 100)
        exe.submit(train_svd, cm10, vocab, 10, 50)
        exe.submit(train_svd, cm5, vocab, 5, 300)
        exe.submit(train_svd, cm5, vocab, 5, 100)
        exe.submit(train_svd, cm5, vocab, 5, 50)
        exe.submit(train_svd, cm2, vocab, 2, 300)
        exe.submit(train_svd, cm2, vocab, 2, 100)
        exe.submit(train_svd, cm2, vocab, 2, 50)
    sleep(1)


def load_brown_w2v(path_to_brown: str) -> List[List[str]]:
    """Extracts sentences from the Brown corpus."""
    sentences = []
    for line in open(path_to_brown, 'r').readlines():
        line = line.strip().lower().replace('``', '').replace(',', '').replace('\'\'', '').replace('.', '')
        tokens = nltk.word_tokenize(line)
        sentences.append(tokens)  # assuming each line is a sentence
    return sentences


def train_w2v(window_sz: int,
              vec_dimension: int,
              neg_sampl: int,
              path_to_brown: str = DEFAULT_BROWN_PATH):
    sentences = load_brown_w2v(path_to_brown)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vec_dimension,
        window=window_sz,
        negative=neg_sampl,
        min_count=2,
        epochs=100,
        workers=1   # will do multiple training processes in parallel, so set this to 1
    )
    model.save('word2vec-w%d-k%d-n%d.wv' % (window_sz, vec_dimension, neg_sampl))


def train_all_w2v():
    with ProcessPoolExecutor(N_WORKERS) as exe:
        exe.submit(train_w2v, 10, 300, 15)
        exe.submit(train_w2v, 10, 300, 5)
        exe.submit(train_w2v, 10, 300, 1)
        exe.submit(train_w2v, 10, 100, 15)
        exe.submit(train_w2v, 10, 100, 5)
        exe.submit(train_w2v, 10, 100, 1)
        exe.submit(train_w2v, 10, 50, 15)
        exe.submit(train_w2v, 10, 50, 5)
        exe.submit(train_w2v, 10, 50, 1)
        exe.submit(train_w2v, 5, 300, 15)
        exe.submit(train_w2v, 5, 300, 5)
        exe.submit(train_w2v, 5, 300, 1)
        exe.submit(train_w2v, 5, 100, 15)
        exe.submit(train_w2v, 5, 100, 5)
        exe.submit(train_w2v, 5, 100, 1)
        exe.submit(train_w2v, 5, 50, 15)
        exe.submit(train_w2v, 5, 50, 5)
        exe.submit(train_w2v, 5, 50, 1)
        exe.submit(train_w2v, 2, 300, 15)
        exe.submit(train_w2v, 2, 300, 5)
        exe.submit(train_w2v, 2, 300, 1)
        exe.submit(train_w2v, 2, 100, 15)
        exe.submit(train_w2v, 2, 100, 5)
        exe.submit(train_w2v, 2, 100, 1)
        exe.submit(train_w2v, 2, 50, 15)
        exe.submit(train_w2v, 2, 50, 5)
        exe.submit(train_w2v, 2, 50, 1)
    sleep(1)


if __name__ == '__main__':
    train_all_svd()
    train_all_w2v()
