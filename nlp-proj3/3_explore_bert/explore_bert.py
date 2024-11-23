from concurrent.futures import ProcessPoolExecutor
import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import *

PATH_TO_CONLL = 'data/conll.pt'
PATH_TO_SEMEVAL = 'data/semeval.pt'
N_LAYERS = 13
N_PROCESSES = 4


def process_conll_subset(
        subset: List[Dict],
        pos_encoder: LabelEncoder,
        ner_encoder: LabelEncoder) -> Tuple[np.ndarray, List, List]:
    """For Conll dataset, extracts word embeddings and convert string labels to int."""
    all_x = [[] for _ in range(N_LAYERS)]
    y_pos, y_ner = [], []
    for d in subset:
        hidden = d['hidden_states']
        y_pos.extend(pos_encoder.transform(d['pos_labels']))
        y_ner.extend(ner_encoder.transform(d['ner_labels']))
        for tidx in d['word_token_indices']:
            for i in range(N_LAYERS):
                all_x[i].append(hidden[i][tidx].mean(0).numpy())
    return np.asarray(all_x), y_pos, y_ner


def explore_layer(
        x_train: np.ndarray,
        y_train: List,
        x_val: np.ndarray,
        y_val: List,
        max_iter: int) -> float:
    """Trains and evaluates POS or NER for a specific layer."""
    model = LogisticRegression(max_iter=max_iter, class_weight='balanced')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    return f1_score(y_val, y_pred, average='macro')


def layer_worker(
        x_train: np.ndarray,
        y_pos_train: List,
        y_ner_train: List,
        x_val: np.ndarray,
        y_pos_val: List,
        y_ner_val: List) -> Tuple[float, float]:
    """Calculates POS and NER macro F1 scores for a specific layer."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    f1_pos = explore_layer(
        x_train, y_pos_train, x_val, y_pos_val, 2500)
    f1_ner = explore_layer(
        x_train, y_ner_train, x_val, y_ner_val, 2500)
    return f1_pos, f1_ner


def explore_bert_conll(train: List[Dict], val: List[Dict]) -> Tuple[List[float], List[float]]:
    # d = load(path_to_pt, map_location='cpu')
    # train, val = d['train'], d['validation']

    # Gets all possible labels
    pos_classes, ner_classes = set(), set()
    for d in train:
        pos_classes.update(d['pos_labels'])
        ner_classes.update(d['ner_labels'])
    for d in val:
        pos_classes.update(d['pos_labels'])
        ner_classes.update(d['ner_labels'])
    pos_encoder, ner_encoder = LabelEncoder(), LabelEncoder()
    pos_encoder.fit(sorted(pos_classes))
    ner_encoder.fit(sorted(ner_classes))

    # Extracts training data and labels
    x_train, y_pos_train, y_ner_train = process_conll_subset(train, pos_encoder, ner_encoder)
    x_val, y_pos_val, y_ner_val = process_conll_subset(val, pos_encoder, ner_encoder)

    # Explores layers in parallel
    y_axis_pos, y_axis_ner = [], []
    with ProcessPoolExecutor(N_PROCESSES) as exe:
        futures = []
        for i in range(N_LAYERS):
            x_train_i, x_val_i = x_train[i].copy(), x_val[i].copy()
            futures.append(exe.submit(
                layer_worker,
                x_train_i, y_pos_train, y_ner_train,
                x_val_i, y_pos_val, y_ner_val
            ))

        for i in range(N_LAYERS):
            f1_pos, f1_ner = futures[i].result()
            y_axis_pos.append(f1_pos)
            y_axis_ner.append(f1_ner)
    return y_axis_pos, y_axis_ner


def process_semeval_subset(
        subset: List[Dict],
        y_encoder: LabelEncoder) -> Tuple[np.ndarray, List]:
    """For semeval dataset, extracts word embeddings and convert string labels to int."""
    all_x = [[] for _ in range(N_LAYERS)]
    y = []
    for d in subset:
        entity1_h = d['entity1_representations']
        entity2_h = d['entity2_representations']
        y.extend(y_encoder.transform([d['rel_label']]))
        for i in range(N_LAYERS):
            all_x[i].append(np.concatenate(
                (entity1_h[i].numpy(), entity2_h[i].numpy())))
    return np.asarray(all_x), y


def explore_bert_semieval(train: List[Dict], val: List[Dict]) -> List[float]:
    # d = load(path_to_pt, map_location='cpu')
    # train, val = d['train'], d['test']

    # Gets all possible labels
    rel_classes = set()
    for d in train:
        rel_classes.add(d['rel_label'])
    for d in val:
        rel_classes.add(d['rel_label'])
    rel_encoder = LabelEncoder()
    rel_encoder.fit(sorted(rel_classes))

    # Extracts training data and labels
    x_train, y_train = process_semeval_subset(train, rel_encoder)
    x_val, y_val = process_semeval_subset(val, rel_encoder)

    # Explores layers in parallel
    y_axis = []
    with ProcessPoolExecutor(N_PROCESSES) as exe:
        futures = []
        for i in range(N_LAYERS):
            x_train_i, x_val_i = x_train[i].copy(), x_val[i].copy()
            futures.append(exe.submit(
                explore_layer,
                x_train_i, y_train, x_val_i, y_val, 2000
            ))

        for i in range(N_LAYERS):
            y_axis.append(futures[i].result())
    return y_axis


def plot(y_axis_pos: List[float], y_axis_ner: List[float], y_axis_rel: List[float] = None):
    # Plots results
    fig = plt.figure()
    x_axis = list(range(len(y_axis_pos)))
    plt.plot(x_axis, y_axis_pos, label='POS')
    plt.plot(x_axis, y_axis_ner, label='NER')
    if y_axis_rel:
        plt.plot(x_axis, y_axis_rel, label='REL')
    plt.xlabel('BERT layer index')
    plt.ylabel('Macro-F1')
    plt.title('F1 Scores by Layer')
    plt.legend()
    plt.show()
    fig.savefig('part3.png', dpi=fig.dpi)


def main():
    from torch import load  # import torch at the top will cause multiprocessing to crash
    
    conll = load(PATH_TO_CONLL, map_location='cpu')
    y_axis_pos, y_axis_ner = explore_bert_conll(conll['train'], conll['validation'])
    # y_axis_pos = [0.6757062331301705, 0.6941341111614534, 0.7164766747759299, 0.7223616153772693, 0.7088489250103476,
    #               0.7410306602751647, 0.7423879812143205, 0.7615154998472248, 0.7405918258490469, 0.7394831236608732,
    #               0.7296162951519618, 0.7259901333566747, 0.707969306423888]
    # y_axis_ner = [0.6131580283158871, 0.6690930761327885, 0.7043461098472059, 0.724501729313433, 0.7595122816702263,
    #               0.7759664294386509, 0.7873428161510185, 0.7837098284999489, 0.793042798299728, 0.8020675865724802,
    #               0.8061006066902896, 0.7953868716978475, 0.8006855861880663]

    semieval = load(PATH_TO_SEMEVAL, map_location='cpu')
    y_axis_rel = explore_bert_semieval(semieval['train'], semieval['test'])
    plot(y_axis_pos, y_axis_ner, y_axis_rel)


if __name__ == '__main__':
    main()
