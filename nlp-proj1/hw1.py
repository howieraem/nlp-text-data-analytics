"""Author: J Lin"""
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from features import *


class Model:
    def __init__(self, feat_extractor: FeatExtractor, clf: LogisticRegression, model_type: str):
        self.X_train, self.X_test = feat_extractor.get_features(model_type)
        self.y_train, self.y_test = feat_extractor.get_labels()
        self.y_pred = None
        self.clf = clf

    def train(self):
        self.clf.fit(self.X_train, self.y_train)

    def pred(self):
        self.y_pred = self.clf.predict(self.X_test)

    def eval(self, subset_row_idxs):
        assert self.y_pred is not None, "run() must be called before model evaluation!"
        if subset_row_idxs is not None and len(subset_row_idxs):
            y_test, y_pred = self.y_test[subset_row_idxs], self.y_pred[subset_row_idxs]
        else:
            y_test, y_pred = self.y_test, self.y_pred
        return accuracy_score(y_test, y_pred)

    def run(self):
        self.train()
        self.pred()
        return self


def main(train_file_path: str,
         test_file_path: str,
         usr_data_path: str,
         mdl_type: str,
         lex_dir_path: str,
         out_file_path: str):
    """
    Main function to load data, process data, train model and predict result.

    Args:
        train_file_path: full path to the training file
        test_file_path: full path to the evaluation file
        usr_data_path: full path to the user data file
        mdl_type: the name of the model to train and evaluate
        lex_dir_path: full path to the directory containing the lexica
        out_file_path: full path to the file to write model predictions

    Returns:
        None

    """
    target_names = ['Pro', 'Con']

    dfs_train, dfs_test, lex_c, lex_v, usr_data = load_data(
        train_file_path, test_file_path, lex_dir_path, usr_data_path)
    df_train, df_train_r, df_train_nr = dfs_train
    df_test, df_test_r, df_test_nr = dfs_test

    idx_test_r = df_test_r.index
    idx_test_nr = df_test_nr.index

    # Feature extractors for religious and non-religious data respectively
    fe_r = FeatExtractor(df_train, df_test, lex_c, lex_v, usr_data)
    fe_r.set_feature_types('conno', ('length', 'pronoun'), ('big_issues', 'politic'))
    fe_nr = FeatExtractor(df_train, df_test, lex_c, lex_v, usr_data)
    fe_nr.set_feature_types('vad', ('length', 'pronoun'), ('big_issues', 'friend'))

    # Models for religious and non-religious data respectively
    model1 = Model(
        fe_r,
        LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.97, max_iter=35, random_state=0, tol=1e-8),
        mdl_type
    ).run()
    model2 = Model(
        fe_nr,
        LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.97, max_iter=35, random_state=0, tol=1e-8),
        mdl_type
    ).run()

    # if mdl_type == 'Ngram+Lex+Ling':
    #     model1, model2 = model2, model1

    # Concatenate separate predictions and write to file
    pred_r = model1.y_pred
    pred_nr = model2.y_pred
    pred_out = [''] * len(df_test)
    for i in idx_test_r:
        pred_out[i] = target_names[pred_r[i]]
    for i in idx_test_nr:
        pred_out[i] = target_names[pred_nr[i]]
    assert all(s for s in pred_out), 'Some predictions are missing!'
    with open(out_file_path, 'w+') as f:
        for line in pred_out:
            f.write(line + '\n')

    # Print evaluation results
    acc_r = model1.eval(idx_test_r)
    print("Religious accuracy: ", round(acc_r, 4))
    acc_nr = model2.eval(idx_test_nr)
    print("Non-religious accuracy: ", round(acc_nr, 4))
    acc = (len(df_test_r) * acc_r + len(df_test_nr) * acc_nr) / len(df_test)
    print("Overall accuracy: ", round(acc, 4))
    print("Overall accuracy (2nd check): ",
          round(sum(s == l for s, l in zip(pred_out, df_test['winner'])) / len(df_test), 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--user_data', dest='user_data', required=True,
                        help='Full path to the user data file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    parser.add_argument('--outfile', dest='outfile', required=True,
                        help='Full path to the file we will write the model predictions')
    args = parser.parse_args()

    main(args.train, args.test, args.user_data, args.model, args.lexicon_path, args.outfile)
