"""Build a sentiment analysis model

Sentiment analysis can be modeled as a binary text classification problem.
Here, we fit a linear classifier on features extracted from movie reviews
in order to predict whether the opinion expressed by the author is positive or negative.

"""

from scipy.sparse import csr_matrix, vstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from nltk import word_tokenize

np.random.seed(0)


class CountVectorizer:
    def __init__(self, n_gram_range):
        self.n_gram_range = n_gram_range

    def fit(self, all_documents, _):
        # 1. lowercase each document
        # 2. tokenize each document using nltk's word_tokenize
        # 3. extract all n-grams (unigrams, bigrams, etc as 
        #    specified by self.n_gram_range) from these tokens;
        #    represent each n-gram as a tuple (eg, ('dog',), ('cat', 'fight'))
        # 4. collect all unique n-grams across all the documents, sort
        #    them and save them as a dictionary in self.vocabulary_
        #    where the keys are the n-grams and the values are their sorted indices
        self.vocabulary_ = dict()
        tmp = set()
        for doc in all_documents:
            tokens = tuple(word_tokenize(doc.lower()))
            for i in range(len(tokens)):
                for n in range(self.n_gram_range[0], self.n_gram_range[1] + 1):
                    if i + n > len(tokens):
                        break
                    gram = tokens[i:i + n]
                    tmp.add(gram)
        for i, k in enumerate(sorted(tmp)):
            self.vocabulary_[k] = i
        return self

    def transform(self, all_documents):
        # 1. lowercase each document
        # 2. tokenize each document using nltk's word_tokenize
        # 3. extract all n-grams (unigrams, bigrams, etc as 
        #    specified by self.n_gram_range) from these tokens;
        #    represent each n-gram as a tuple (eg, ('dog',), ('cat', 'fight'))
        # 4. represent each document's n-grams as a sparse vector whose indices correspond
        #    to counts of n-grams in the sorted vocabulary_; Google "bag-of-words" if that's unclear!;
        #    if an n-gram isn't in the vocabulary_ (it wasn't in the training set), skip it!
        # 5. return a single vector of shape (# documents, # of n-grams in vocabulary)
        all_document_representations = []
        for document in all_documents:
            document_representation = np.zeros(len(self.vocabulary_), dtype=int)

            document = document.lower()
            tokens = tuple(word_tokenize(document))
            for i in range(len(tokens)):
                for n in range(self.n_gram_range[0], self.n_gram_range[1] + 1):
                    if i + n > len(tokens):
                        break
                    gram = tokens[i:i + n]
                    if gram in self.vocabulary_:
                        document_representation[self.vocabulary_[gram]] += 1

            all_document_representations.append(csr_matrix(document_representation))

        return vstack(all_document_representations)


# if the accuracy is undefined, return None
def calculate_accuracy(y_test, y_predicted):
    n_correct = 0
    for yt, yp in zip(y_test, y_predicted):
        n_correct += (yt == yp)
    return n_correct / len(y_test)


# calculate class-based precisions, returning a tuple 
# of length 2: (class 0 precision, class 1 precision)
# if a precision is undefined, return None for that class
def calculate_precisions(y_test, y_predicted):
    tp0 = tp1 = fp0 = fp1 = 0
    for yt, yp in zip(y_test, y_predicted):
        tp0 += not (yt or yp)   # yt == 0 and yp == 0
        tp1 += yt and yp        # yt == 1 and yp == 1
        fp0 += yt and not yp    # yt == 1 and yp == 0
        fp1 += yp and not yt    # yt == 0 and yp == 1
    return tp0 / (tp0 + fp0), tp1 / (tp1 + fp1)


# calculate class-based recalls, returning a tuple 
# of length 2: (class 0 recall, class 1 recall)
# if a recall is undefined, return None for that class
def calculate_recalls(y_test, y_predicted):
    tp0 = fn0 = tp1 = fn1 = 0
    for yt, yp in zip(y_test, y_predicted):
        tp0 += not (yt or yp)   # yt == 0 and yp == 0
        tp1 += yt and yp        # yt == 1 and yp == 1
        fn0 += yp and not yt    # yt == 0 and yp == 1
        fn1 += yt and not yp    # yt == 1 and yp == 0
    return tp0 / (tp0 + fn0), tp1 / (tp1 + fn1)


if __name__ == "__main__":
    # Collect the training data

    movie_reviews_data_folder = "data/txt_sentoken/"
    dataset = load_files(
        movie_reviews_data_folder, shuffle=False, encoding='utf-8')
    print("n_samples: %d" % len(dataset.data))

    # Split the dataset into training and test subsets

    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # Build a vectorizer / classifier pipeline that uses unigrams, bigrams and trigrams

    # A simplified version of scikit-learn's CountVectorizer above (line 28) 
    clf = Pipeline([
        ('vect', CountVectorizer(n_gram_range=(1, 3))),
        ('clf', KNeighborsClassifier(n_neighbors=10)),
    ])

    # Train the classifier on the training set

    clf.fit(docs_train, y_train)

    # Predict the outcome on the testing set and store it 

    y_predicted = clf.predict(docs_test)

    # Get the probabilities we'll need for the precision-recall curve

    y_probs = clf.predict_proba(docs_test)[:, 1]

    # Evaluate our model

    # Implement calculate_accuracy, calculate_precisions and calculate_recalls
    print("accuracy: %0.4f" % calculate_accuracy(y_test, y_predicted))
    print("precisions: %0.4f, %0.4f" % calculate_precisions(y_test, y_predicted))
    print("recalls: %0.4f, %0.4f" % calculate_recalls(y_test, y_predicted))

    # Calculate and plot the precision-recall curve
    # Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y_test, y_probs)

    plt.figure()
    plt.step(r, p)
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])  # slightly greater than 1 to avoid overlap
    plt.grid()
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("Precision-Recall Curve")
    plt.savefig("plot.png")
    # plt.show()
