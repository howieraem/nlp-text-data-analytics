from concurrent.futures import ProcessPoolExecutor
import argparse
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time
import torch
from transformers import BertModel, BertTokenizer
from typing import *


LABELS = ['F', 'T']
USE_GPU = True  # this code will work with an NVIDIA GPU with >= 2.5 GB RAM available
DEVICE = torch.device('cuda:0') if USE_GPU and torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE_BERT = 256	 # lower this if your GPU ran out of memory


def get_wic_subset(data_dir: str):
    wic = []
    split = data_dir.strip().split('/')[-1]
    with open(os.path.join(data_dir, '%s.data.txt' % split), 'r', encoding='utf-8') as datafile, \
        open(os.path.join(data_dir, '%s.gold.txt' % split), 'r', encoding='utf-8') as labelfile:
        for (data_line, label_line) in zip(datafile.readlines(), labelfile.readlines()):
            word, _, word_indices, sentence1, sentence2 = data_line.strip().split('\t')
            sentence1_word_index, sentence2_word_index = word_indices.split('-')
            sentence1_words = sentence1.split(' ')
            sentence2_words = sentence2.split(' ')
            label = LABELS.index(label_line.strip())
            wic.append({
                'word': word,
                'sentence1_word_index': int(sentence1_word_index),
                'sentence2_word_index': int(sentence2_word_index),
                'sentence1': sentence1,
                'sentence2': sentence2,
                'sentence1_words': sentence1_words,
                'sentence2_words': sentence2_words,
                'sentence1_length': len(sentence1_words),
                'sentence2_length': len(sentence2_words),
                'label': label
            })
    return wic


def generate_raw_embeddings(
        sentences: List[str],
        tokenizer: BertTokenizer,
        bert: BertModel) -> torch.Tensor:
    """Generates embeddings for a list of sentences."""
    tokenized = tokenizer(sentences, padding=True, return_tensors='pt').to(DEVICE)
    input_ids = tokenized['input_ids'].split(BATCH_SIZE_BERT)
    token_type_ids = tokenized['token_type_ids'].split(BATCH_SIZE_BERT)
    attention_mask = tokenized['attention_mask'].split(BATCH_SIZE_BERT)
    raw_outputs = []
    for ipi, tti, am in zip(input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            out = bert(ipi, tti, am)
        raw_outputs.append(out.last_hidden_state.cpu())
    return torch.cat(raw_outputs, dim=0)


def generate_embeddings(
        wic: List[Dict],
        tokenizer: BertTokenizer,
        bert: BertModel,
        mean_window: int = 5) -> np.ndarray:
    """Generates embeddings from WIC training/evaluation set."""
    # Collect all the sentences for batch computation
    sentences1, sentences2 = [], []
    for d in wic:
        sentences1.append(d['sentence1'])
        sentences2.append(d['sentence2'])

    # Generate raw embeddings
    raw_outputs1 = generate_raw_embeddings(sentences1, tokenizer, bert)
    raw_outputs2 = generate_raw_embeddings(sentences2, tokenizer, bert)

    # Post-processing in pairs of sentences
    outputs = []
    for data_dict, raw1, raw2 in zip(wic, raw_outputs1, raw_outputs2):
        w1i, w2i = data_dict['sentence1_word_index'], data_dict['sentence2_word_index']
        l1, l2 = data_dict['sentence1_length'], data_dict['sentence2_length']
        embed1 = raw1[max(w1i - mean_window, 0):min(w1i + mean_window + 1, l1)]
        embed2 = raw2[max(w2i - mean_window, 0):min(w2i + mean_window + 1, l2)]
        outputs.append(torch.cat((embed1, embed2), 0).mean(0))
    outputs = torch.stack(outputs).numpy()
    return outputs


def main(train_dir: str,
         eval_dir: str,
         out_file: str):
    t = time.time()

    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert = BertModel.from_pretrained('bert-base-cased')
    bert = bert.eval()
    bert = bert.to(DEVICE)

    # Load datasets and generate embeddings
    wic_train = get_wic_subset(train_dir)
    wic_val = get_wic_subset(eval_dir)
    if USE_GPU:
        # GPU is fast enough, so multiprocessing is not used here.
        X_train = generate_embeddings(wic_train, tokenizer, bert)
        X_val = generate_embeddings(wic_val, tokenizer, bert)
        y_train = [d['label'] for d in wic_train]
        y_val = [d['label'] for d in wic_val]
    else:
        # Make full use of CPU by processing training and evaluation sets
        # at the same time.
        with ProcessPoolExecutor(2) as exe:
            future1 = exe.submit(generate_embeddings, wic_train, tokenizer, bert)
            future2 = exe.submit(generate_embeddings, wic_val, tokenizer, bert)
            y_train = [d['label'] for d in wic_train]
            y_val = [d['label'] for d in wic_val]
            X_train, X_val = future1.result(), future2.result()

    # Binary classification with SVM
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    clf = SVC(tol=1e-8)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    # Report results
    print('Accuracy:', accuracy_score(y_val, y_pred))
    print('Time elapsed:', time.time() - t, 'seconds')
    with open(out_file, 'w+') as f:
        for y in y_pred:
            f.write(LABELS[y] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a classifier to recognize words in context (WiC).'
    )
    parser.add_argument(
        '--train-dir',
        dest='train_dir',
        required=True,
        help='The absolute path to the directory containing the WiC train files.'
    )
    parser.add_argument(
        '--eval-dir',
        dest='eval_dir',
        required=True,
        help='The absolute path to the directory containing the WiC eval files.'
    )
    # Write your predictions (F or T, separated by newlines) for each evaluation
    # example to out_file in the same order as you find them in eval_dir.  For example:
    # F
    # F
    # T
    # where each row is the prediction for the corresponding line in eval_dir.
    parser.add_argument(
        '--out-file',
        dest='out_file',
        required=True,
        help='The absolute path to the file where evaluation predictions will be written.'
    )
    args = parser.parse_args()
    main(args.train_dir, args.eval_dir, args.out_file)
