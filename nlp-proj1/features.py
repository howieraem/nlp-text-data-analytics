"""Author: J Lin"""
import json
from nltk import download, word_tokenize
import numpy as np
import os.path as osp
import pandas as pd
from scipy.sparse import *
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple


__all__ = ['FeatExtractor', 'load_data']

# Load vocabularies
download('punkt')

# Constant
N_BIG_ISSUES = 48


def load_df(jsonl_path: str, split_category: str = "Religion") -> tuple:
    """
    Loads data frame from a *.jsonl file and splits it into two data frames by category.

    Args:
        jsonl_path: path to the train or test *.jsonl file.
        split_category:

    Returns:
        tuple of data frames: (full data frame, data frame of category, data frame of other categories)

    """
    df = pd.read_json(jsonl_path, lines=True)
    mask = df['category'] == split_category
    return df, df[mask], df[~mask]


def load_lex_conno(lexicon_path: str) -> dict:
    res = {}
    with open(lexicon_path, 'r') as f:
        for line in f:
            if not line:
                continue
            word_info, word_cls = line.strip().split(',')
            word, _ = word_info.split('_')
            if word_cls == 'positive':
                res[word] = 0
            elif word_cls == 'negative':
                res[word] = 1
            else:
                res[word] = 2
    return res


def load_lex_vad(lexicon_path: str) -> dict:
    res = {}
    with open(lexicon_path, 'r') as f:
        for line in f:
            if not line:
                continue
            info = line.split()
            res[' '.join(info[:-3])] = float(info[-3]), float(info[-2]), float(info[-1])
    return res


def load_user_data(user_data_path: str) -> dict:
    with open(user_data_path, 'r') as f:
        d = json.load(f)
    return d


def load_data(train_file_path: str,
              test_file_path: str,
              lex_dir_path: str,
              usr_data_path: str) -> tuple:
    return (load_df(train_file_path),
            load_df(test_file_path),
            load_lex_conno(osp.join(lex_dir_path, 'connotation_lexicon_a.0.1.csv')),
            load_lex_vad(osp.join(lex_dir_path, 'NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt')),
            load_user_data(usr_data_path))


def eval_big_issues_dict(big_issues: dict):
    res = np.zeros(N_BIG_ISSUES, dtype=np.float32)
    keys = sorted(big_issues.keys())
    for i, k in enumerate(keys):
        opinion = big_issues[k]
        if opinion == 'Pro':
            continue    # keep 0
        elif opinion == 'Con':
            res[i] = 1
        else:
            res[i] = 0.5
    return res


class FeatExtractor:
    MODAL_VERBS = {
        'can', 'could', 'may', 'might', 'will', 'would', 'shall', 'should', 'must', 'ought', 'dare'
    }

    PRONOUNS_FIRST = {
        'I', 'we', 'me', 'us', 'my', 'mine', 'our', 'ours', 'myself', 'ourselves'
    }

    # PRONOUNS_SECOND = {
    #     'you', 'your', 'yours', 'yourself', 'yourselves'
    # }

    PRONOUNS_THIRD = {
        'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'hers', 'its', 'their', 'theirs',
        'himself', 'herself', 'themselves', 'itself'
    }

    def __init__(self,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame,
                 lex_conno: dict,
                 lex_vad: dict,
                 user_data: dict):
        """
        Constructor of feature extractor.

        Args:
            df_train: Data frame of the training data.
            df_test: Data frame of the test data.
            lex_conno: Dictionary containing the Connotation lexicon.
            lex_vad: Dictionary containing the NRC-VAD (English) lexicon.
            user_data: Dictionary containing information of debaters.
        """
        self.df_train = df_train
        self.df_test = df_test
        self.lex_conno = lex_conno
        self.lex_vad = lex_vad
        self.user_data = user_data

        self.vectorizer_pro = TfidfVectorizer(ngram_range=(1, 3), max_df=0.96, min_df=0.056, dtype=np.float32)
        self.vectorizer_con = TfidfVectorizer(ngram_range=(1, 3), max_df=0.96, min_df=0.056, dtype=np.float32)

        self.aux_feats_train = dict()
        self.aux_feats_test = dict()
        self.selected_lex_feat = None
        self.selected_ling_feats = (None, None)
        self.selected_user_feats = (None, None)

        self.X_train_ngram: coo_matrix = None
        self.X_test_ngram: coo_matrix = None
        self.X_train_lex: np.ndarray = None
        self.X_test_lex: np.ndarray = None
        self.X_train_ling: np.ndarray = None
        self.X_test_ling: np.ndarray = None
        self.X_train_user: np.ndarray = None
        self.X_test_user: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None

    def _process_df(self, df: pd.DataFrame) -> Tuple[dict, np.ndarray]:
        """
        Processes data frame and returns raw features.

        Args:
            df: Data frame to be processed which must have the same format as specified in homework description PDF.

        Returns:
            A tuple: (A dictionary with feature names as keys and raw features as values,
                      labels stored in a numpy array)

        """
        n = len(df)

        """Feature placeholders for all documents."""
        # Corpus for ngram feature
        corpus_pro, corpus_con = [], []

        # Lex features
        conno = np.zeros((n, 2), dtype=np.float32)
        vad_pro = np.zeros((n, 3), dtype=np.float32)
        vad_con = np.zeros((n, 3), dtype=np.float32)

        # Ling features
        length = np.zeros((n, 1), dtype=np.float32)
        modal_verb = np.zeros((n, 1), dtype=np.float32)
        exclaim = np.zeros((n, 1), dtype=np.float32)
        pronoun = np.zeros((n, 2), dtype=np.float32)

        # User features
        big_issues = np.zeros((n, N_BIG_ISSUES), dtype=np.float32)
        politic = np.zeros((n, 1), dtype=np.float32)
        friend = np.zeros((n, 1), dtype=np.float32)
        gender = np.zeros((n, 1), dtype=np.float32)

        # Label
        label = np.zeros(n, dtype=np.int)

        for i, row in df.iterrows():
            """Feature placeholders for the current document."""
            # ngram
            doc_pro, doc_con = [], []

            # lex
            doc_vad_pro, doc_vad_con = [], []
            doc_conno_pro, doc_conno_con = np.zeros(3), np.zeros(3)

            # ling
            len_pro = len_con = 0
            modal_verb_pro = modal_verb_con = 0
            exclaim_pro = exclaim_con = 0
            pronoun_pro, pronoun_con = np.zeros(3), np.zeros(3)

            # user
            pro, con = row['pro_debater'], row['con_debater']
            pro_info, con_info = self.user_data[pro], self.user_data[con]
            big_issues[i] = (eval_big_issues_dict(pro_info['big_issues_dict']) ==
                             eval_big_issues_dict(con_info['big_issues_dict']))
            politic[i] = pro_info['political_ideology'] == con_info['political_ideology']
            friend[i] = pro in con_info['friends'] and con in pro_info['friends']
            gender[i] = pro_info['gender'] == con_info['gender']

            for debate_round in row['rounds']:
                for side in debate_round:
                    text = side['text']
                    tokens = word_tokenize(text.lower())
                    if side['side'] == 'Pro':
                        doc_pro.append(text)
                        len_pro += len(tokens)
                        for token in tokens:
                            # Connotation
                            if token in self.lex_conno:
                                doc_conno_pro[self.lex_conno[token]] += 1
                            # VAD
                            if token in self.lex_vad:
                                doc_vad_pro.append(self.lex_vad[token])
                            # Modal verbs
                            if token in self.MODAL_VERBS:
                                modal_verb_pro += 1
                            # Exclamations
                            if token == '!':
                                exclaim_pro += 1
                            # 1st-person pronouns
                            if token in self.PRONOUNS_FIRST:
                                pronoun_pro[0] += 1
                            # 2nd-person pronouns (ignored)
                            # if token in self.PRONOUNS_SECOND:
                            #     pronoun_pro[1] += 1
                            # 3rd-person pronouns
                            if token in self.PRONOUNS_THIRD:
                                pronoun_pro[2] += 1
                    else:
                        doc_con.append(text)
                        len_con += len(tokens)
                        for token in tokens:
                            # Connotation
                            if token in self.lex_conno:
                                doc_conno_con[self.lex_conno[token]] += 1
                            # VAD
                            if token in self.lex_vad:
                                doc_vad_con.append(self.lex_vad[token])
                            # Modal verbs
                            if token in self.MODAL_VERBS:
                                modal_verb_con += 1
                            # Exclamations
                            if token == '!':
                                exclaim_con += 1
                            # 1st-person pronouns
                            if token in self.PRONOUNS_FIRST:
                                pronoun_con[0] += 1
                            # 2nd-person pronouns (ignored)
                            # if token in self.PRONOUNS_SECOND:
                            #     pronoun_con[1] += 1
                            # 3rd-person pronouns
                            if token in self.PRONOUNS_THIRD:
                                pronoun_con[2] += 1

            """Feature post-processing for the current document."""
            # ngram
            corpus_pro.append(''.join(doc_pro))
            corpus_con.append(''.join(doc_con))

            # lex
            conno[i] = doc_conno_pro[[1, 2]] > doc_conno_con[[1, 2]]    # positive word count ignored
            if len(doc_vad_pro):
                vad_pro[i] = np.asarray(doc_vad_pro, dtype=np.float32).mean(axis=0)
            if len(doc_vad_con):
                vad_con[i] = np.asarray(doc_vad_con, dtype=np.float32).mean(axis=0)

            # Ling features
            length[i] = len_pro > len_con
            modal_verb[i] = modal_verb_pro > modal_verb_con
            exclaim[i] = exclaim_pro > exclaim_con
            pronoun[i] = pronoun_pro[[0, 2]] > pronoun_con[[0, 2]]  # 2nd-person pronouns ignored
            # print(row['id'], doc_conno_pro[[1, 2]], doc_conno_con[[1, 2]], vad_pro[i], vad_con[i], row['winner'])

            # Label
            label[i] = (row['winner'] == 'Con')  # Pro 0, Con 1

        return {
            'corpus_pro': corpus_pro,
            'corpus_con': corpus_con,
            'conno': conno,
            'vad': np.concatenate((vad_pro, vad_con), axis=1),
            'length': length,
            'modal_verb': modal_verb,
            'exclaim': exclaim,
            'pronoun': pronoun,
            'big_issues': big_issues,
            'politic': politic,
            'friend': friend,
            'gender': gender
        }, label

    def set_feature_types(self, lex_type: str, ling_types: tuple, user_types: tuple):
        """Defines feature sets for the extractor. Must be called before get_features()."""
        self.selected_lex_feat = lex_type
        self.selected_ling_feats = ling_types
        self.selected_user_feats = user_types
        self._extract_features(True)
        self._extract_features(False)

    def _extract_features(self, is_train: bool):
        """Extracts features from the training and the test data frames."""
        assert self.selected_lex_feat, 'Must define a lexicon feature in child class.'
        assert self.selected_ling_feats[0] and self.selected_ling_feats[1], \
            'Must define two linguistic features in child class.'
        assert self.selected_user_feats[0] and self.selected_user_feats[1], \
            'Must define two user features in child class.'
        if is_train:
            aux_feats_train, self.y_train = self._process_df(self.df_train)
            ngrams_pro = self.vectorizer_pro.fit_transform(aux_feats_train['corpus_pro'])
            ngrams_con = self.vectorizer_con.fit_transform(aux_feats_train['corpus_con'])
            self.X_train_ngram = hstack((ngrams_pro, ngrams_con))
            self.X_train_lex = aux_feats_train[self.selected_lex_feat]
            self.X_train_ling = np.concatenate(
                (aux_feats_train[self.selected_ling_feats[0]], aux_feats_train[self.selected_ling_feats[1]]),
                axis=1)
            self.X_train_user = np.concatenate(
                (aux_feats_train[self.selected_user_feats[0]], aux_feats_train[self.selected_user_feats[1]]),
                axis=1)
        else:
            aux_feats_test, self.y_test = self._process_df(self.df_test)
            ngrams_pro = self.vectorizer_pro.transform(aux_feats_test['corpus_pro'])
            ngrams_con = self.vectorizer_con.transform(aux_feats_test['corpus_con'])
            self.X_test_ngram = hstack((ngrams_pro, ngrams_con))
            self.X_test_lex = aux_feats_test[self.selected_lex_feat]
            self.X_test_ling = np.concatenate(
                (aux_feats_test[self.selected_ling_feats[0]], aux_feats_test[self.selected_ling_feats[1]]),
                axis=1)
            self.X_test_user = np.concatenate(
                (aux_feats_test[self.selected_user_feats[0]], aux_feats_test[self.selected_user_feats[1]]),
                axis=1)

    def get_features(self, model_type: str):
        """Returns the training and the test features in a pair of scipy sparse matrices."""
        if model_type == "Ngram":
            train_features = (self.X_train_ngram, )
            test_features = (self.X_test_ngram, )
        elif model_type == "Ngram+Lex":
            train_features = (self.X_train_ngram, self.X_train_lex)
            test_features = (self.X_test_ngram, self.X_test_lex)
        elif model_type == "Ngram+Lex+Ling":
            train_features = (self.X_train_ngram, self.X_train_lex, self.X_train_ling)
            test_features = (self.X_test_ngram, self.X_test_lex, self.X_test_ling)
        elif model_type == "Ngram+Lex+Ling+User":
            train_features = (self.X_train_ngram, self.X_train_lex, self.X_train_ling, self.X_train_user)
            test_features = (self.X_test_ngram, self.X_test_lex, self.X_test_ling, self.X_test_user)
        else:
            raise ValueError(
                "Model type incorrect. Must be one of {'Ngram', 'Ngram+Lex', 'Ngram+Lex+Ling', 'Ngram+Lex+Ling+User'}."
            )
        return hstack(train_features), hstack(test_features)

    def get_labels(self):
        """Returns the training and the test labels."""
        return self.y_train, self.y_test
