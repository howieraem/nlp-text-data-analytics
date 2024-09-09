import argparse
from googleapiclient.discovery import build
import numpy as np
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import traceback


# alpha, beta, and gamma used for Rocchio's algorithm
ALPHA = 1.0
BETA = 0.8
GAMMA = 0.1

# Max number of documents retrieved each iteration
N_RETRIEVED = 10

# Read stopwords.txt
with open('stopwords.txt', 'r') as f:
    stop_words = f.read().split('\n')
STOP_WORDS = set(stop_words)


# Get arguments the user enters to retrieve api_key, engine_id, precision, and query
def get_arguments():
    # Used to limit the choices for the precision entered by the user
    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end

        def __eq__(self, x):
            return self.start <= x <= self.end

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Add required positional arguments
    parser.add_argument('api_key', help='Google Custom Search JSON API Key', metavar='<google api key>')
    parser.add_argument('engine_id', help='Google Custom Search Engine ID', metavar='<google engine id>')
    parser.add_argument('precision', type=float, choices=[Range(0.0, 1.0)], help='Target value for precision@10, a real number between 0 and 1', metavar='<precision>')
    parser.add_argument('query', type=str, help='A list of words in double quotes (e.g. "Milky Way"', metavar='<query>')

    # Parse the arguments
    args = parser.parse_args()
    return args


# Formats the arguments user entered
def pprint_args(api_key, engine_id, query, precision):
    print(
        "Parameters:\n"
        "Client key  = {}\n"
        "Engine key  = {}\n"
        "Query       = {}\n"
        "Precision   = {}".format(api_key, engine_id, query, precision)
    )


# Formats the query result output
def pprint_result(results: List[Dict], i: int):
    ret = results[i]
    print(
        "Result {}\n"
        "[\n"
        " URL: {}\n"
        " Title: {}\n"
        " Summary: {}\n"
        "]".format(i + 1, ret['link'], ret['title'], ret['snippet'])
    )


# Function that runs the Google Search Engine search
def search(query: str, api_key: str, engine_id: str, n_ret: int, **kwargs) -> List[Dict]:
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=engine_id, **kwargs).execute()
    return res['items'][:n_ret]


# Calculates precision based on user relevance
def calc_precision(relevance: List[bool]) -> float:
    return sum(relevance) / len(relevance)


# Returns the results based on feedback user provides
def feedback(query: str, cur_precision: float, target_precision: float) -> bool:
    achieved = (cur_precision >= target_precision)
    if achieved:
        conclusion = "Desired precision reached, done"
    elif cur_precision == 0.0:
        conclusion = "Precision is zero, search stopping"
    else:
        conclusion = "Still below the desired precision of {}".format(target_precision)

    print(
        "======================\n"
        "FEEDBACK SUMMARY\n"
        "Query: {}\n"
        "Precision: {}\n".format(query, cur_precision) +
        conclusion
    )
    return achieved or cur_precision == 0.0


# Check if a string is purely English
def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


# Function to clean up query results
def preprocessing(results, query):
    for i in range(len(results)):
        for j in range(len(results[i])):
            # Make all words lowercase
            results[i][j] = results[i][j].lower()

    # Remove stop words and empty strings
    results = [[word for word in sub if len(word) and is_english(word) and (word not in STOP_WORDS or word in query)]
               for sub in results]
    return results


# Function to convert query and documents to vectors
def get_vectors(query_words: List[str], documents: List[List[str]]) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
    # Calculate document frequency and term frequency
    term_docs = defaultdict(set)
    for i in range(len(documents)):
        for word in documents[i]:
            term_docs[word].add(i)

    doc_freq = {}
    for word, doc_set in term_docs.items():
        doc_freq[word] = len(doc_set)

    # Construct list of unique words in all documents
    vocab = list(doc_freq.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Calculate tf-idf features
    document_vecs = []
    for i in range(len(documents)):
        counter = Counter(documents[i])
        document_vecs.append(np.zeros(len(vocab)))
        for word in documents[i]:
            tf = np.log(1 + counter[word])
            df = doc_freq[word]
            idf = np.log(len(documents) / (df + 1)) + 1
            tfidf = tf * idf
            document_vecs[i][word2idx[word]] = tfidf

    query_vec = np.zeros(len(vocab))
    counter = Counter(query_words)
    for word in query_words:
        if word not in doc_freq:
            continue
        tf = np.log(1 + counter[word])
        df = doc_freq[word]
        idf = np.log(len(documents) / (df + 1)) + 1
        tfidf = tf * idf
        query_vec[word2idx[word]] = tfidf
    return query_vec, document_vecs, vocab


# Allocates document vectors according to user feedback, and then run Rocchio's algorithm to produce a new vector
def refine_vector(query_vec: np.ndarray, document_vecs: List[np.ndarray], relevance: List[bool]) -> np.ndarray:
    relevant_vecs, irrelevant_vecs = [], []
    for i in range(len(document_vecs)):
        if relevance[i]:
            relevant_vecs.append(document_vecs[i])
        else:
            irrelevant_vecs.append(document_vecs[i])

    return rocchio(query_vec, ALPHA, relevant_vecs, BETA, irrelevant_vecs, GAMMA)


# Expands the query given the vector output from Rocchio's algorithm
def expand_query(query_words: List[str], refined_vec: np.ndarray, vocab: List[str]):
    # Expand 2 words at a time
    new_query_words_cnt = len(query_words) + 2
    query_words_set = set(w.lower() for w in query_words)

    # Get the indices of the highest scores
    new_query_top_word_idxs = np.argsort(-refined_vec)[:new_query_words_cnt]

    # Map the indices back to words, only add 2 words that are not in the existing query words
    new_query_words = query_words[:]    # make a copy
    cnt = 0
    for wi in new_query_top_word_idxs:
        new_query_word = vocab[wi]
        if new_query_word not in query_words_set:
            cnt += 1
            new_query_words.append(new_query_word)
            if cnt == 2:
                break
    print(f"Augmenting by: {new_query_words[-2]} {new_query_words[-1]}")

    # Rearrange ordering of the new set of query words, along with the old ones
    # Dictionary to sort the words that are of the highest rank
    tmp = {}
    # Array to store a temporary array with query words all in lower case for matching
    test = []
    for i in range(len(new_query_words)):
        test.append(new_query_words[i].lower())

    for i in range(len(refined_vec)):
        if vocab[i] in test:
            tmp[vocab[i]] = refined_vec[i]

    final_dict = {}
    for i in range(len(new_query_words)):
        value = tmp.get(new_query_words[i].lower())
        final_dict[new_query_words[i]] = value

    final_dict = {k: v for k, v in sorted(final_dict.items(), key=lambda item: item[1], reverse=True)}
    reordered_query = list(final_dict.keys())
    return " ".join(reordered_query)


# Rocchio's algorithm
def rocchio(
        q_vec: np.ndarray,
        alpha: float,
        relevant_vecs: List[np.ndarray],
        beta: float,
        irrelevant_vecs: List[np.ndarray],
        gamma: float) -> np.ndarray:
    relevant_vec = np.mean(relevant_vecs, axis=0) if len(relevant_vecs) else np.zeros_like(q_vec)
    irrelevant_vec = np.mean(irrelevant_vecs, axis=0) if len(irrelevant_vecs) else np.zeros_like(q_vec)
    return alpha * q_vec + beta * relevant_vec - gamma * irrelevant_vec


def main():
    # Retrieve user arguments
    args = get_arguments()
    # Remove unnecessary puncutation from user query argument
    query = re.sub(r'[^\w\s]', '', args.query)
    iteration = 0

    while True:
        try:
            # Search on Google
            pprint_args(args.api_key, args.engine_id, query, args.precision)
            documents = []
            res = search(query, args.api_key, args.engine_id, N_RETRIEVED)
            n_res = min(N_RETRIEVED, len(res))
            if not n_res or (iteration == 0 and n_res < 10):
                return
            iteration += 1

            relevance = [False] * n_res
            print("Google Search Results:\n======================")
            for i in range(n_res):
                pprint_result(res, i)
                ret = res[i]
                # documents.append(' '.join([ret['title'], ret['snippet']]).split())
                documents.append(re.sub(r'[^A-Za-z]', ' ', ret['snippet']).split())

                # Asks the user about relevance
                while True:
                    is_relevant = input('Relevant (Y/N)?').lower()
                    if is_relevant == 'y' or is_relevant == 'yes':
                        relevance[i] = True
                        break
                    elif is_relevant == 'n' or is_relevant == 'no':
                        break
                    else:
                        print('Please only enter Y/N (y/n)')

            # Checks if precision has achieved target
            cur_precision = calc_precision(relevance)
            if feedback(query, cur_precision, args.precision):
                break

            # Query Expansion
            documents = preprocessing(documents, query.lower().split(' '))
            query_words = query.strip().split()
            q_vec, d_vecs, vocab = get_vectors(query_words, documents)
            print('Indexing results...')
            new_q_vec = refine_vector(q_vec, d_vecs, relevance)
            print('Indexing results...')
            query = expand_query(query_words, new_q_vec, vocab)
        except KeyboardInterrupt:
            return
        except:
            print(traceback.print_exc())
            return


if __name__ == '__main__':
    main()
