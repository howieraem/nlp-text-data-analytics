import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Ignore HTTPS warning

import argparse
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import re
import requests
import spacy
from spacy_help_functions import extract_relations
from spanbert import SpanBERT
from typing import List, Dict, Tuple
import traceback


# NLP resources
NLP = spacy.load("en_core_web_lg")
SPAN_BERT = SpanBERT("./pretrained_spanbert")

# Relation types and corresponding entities
RELATIONS = {
    1: 'per:schools_attended',
    2: 'per:employee_of',
    3: 'per:cities_of_residence',
    4: 'org:top_members/employees'
}

ENTITIES = {
    'per:schools_attended': ('PERSON', 'ORGANIZATION', ),
    'per:employee_of': ('PERSON', 'ORGANIZATION',),
    'per:cities_of_residence': ('PERSON', 'LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY'),
    'org:top_members/employees': ('ORGANIZATION', 'PERSON'),
}

# Max number of documents retrieved each iteration
N_RETRIEVED = 10

# Webpage Loading Timeout
TIMEOUT = 20

# Maximum characters to process for each webpage document
MAX_CHARS = 20000


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
    parser.add_argument(
        'r',
        help='The relation to extract: 1 is for Schools_Attended, 2 is for Work_For, 3 is for Live_In, and 4 is for Top_Member_Employees',
        metavar='<r>',
        type=int,
        choices=[1, 2, 3, 4]
    )
    parser.add_argument(
        't',
        type=float,
        choices=[Range(0.0, 1.0)],
        help='The extraction confidence threshold, which is the minimum extraction confidence that we request for the tuples in the output',
        metavar='<t>'
    )
    parser.add_argument(
        'q',
        type=str,
        help='A seed query which is a list of words in double quotes corresponding to a plausible tuple for the relation to extract',
        metavar='<q>'
    )
    parser.add_argument(
        'k',
        help='The number of tuples that we request in the output',
        metavar='<k>',
        type=int
    )

    # Parse the arguments
    args = parser.parse_args()
    assert args.k > 0, "<k> must be greater than 0!"
    return args


# Formats the arguments user entered
def pprint_args(api_key, engine_id, relation, threshold, query, n_tuples):
    print(
        "____\n"
        "Parameters:\n"
        "Client key  = {}\n"
        "Engine key  = {}\n"
        "Relation    = {}\n"
        "Threshold   = {}\n"
        "Query       = {}\n"
        "# of Tuples = {}\n"
        "Loading necessary libraries; This should take a minute or so ...".format(api_key, engine_id, relation, threshold, query, n_tuples)
    )


# Formats the query result output
def pprint_results(relation: str, results: Dict[Tuple[str, str, str], float]):
    print(f"================== ALL RELATIONS for {relation} ( {len(results)} ) =================\n")
    for (subj, _, obj), conf in results.items():
        print(f"Confidence: {conf}\t\t\t| Subject: {subj}\t\t\t| Object: {obj}")


# Function that runs the Google Search Engine search and returns a list of URLs
def search(query: str, api_key: str, engine_id: str, n_ret: int, **kwargs) -> List[str]:
    service = build("customsearch", "v1", developerKey=api_key)
    raw_res = service.cse().list(q=query, cx=engine_id, **kwargs).execute()['items'][:n_ret]
    return [ret['link'] for ret in raw_res]


# Get text from URL
def get_webpage_text(url: str, timeout: float = TIMEOUT, max_chars: int = MAX_CHARS) -> str:
    print('\tFetching text from url ...')
    try:
        r = requests.get(url, verify=False, timeout=timeout)
    except requests.exceptions.RequestException:
        return ''
    soup = BeautifulSoup(r.content, features="html.parser")
    raw_p = [''.join(s.find_all(text=True)) for s in soup.find_all('p')]
    raw_lists = [''.join(p.find_all(text=True)).replace('\n',' ').replace('\t', ' ') for p in soup.find_all('li')]

    len_p = len(raw_p)
    len_lists = len(raw_lists)
    max_len = max(len_p, len_lists)
    
    raw_text = []
    for i in range(max_len):
        if i < len_p:
            raw_text.append(raw_p[i])
        if i < len_lists:
            raw_text.append(raw_lists[i])            

    text = " ".join(raw_text)
    # If text is greater than threshold, only return up to the threshold
    if len(text) > max_chars:
        print(f'\tTrimming webpage content from {len(text)} to {max_chars} characters')
        print(f'\tWebpage length (num characters): {max_chars}')
        return text[:max_chars]
    print(f'\tWebpage length (num characters): {len(text)}')
    return text


# Extract relations from text given entities_of_interest
def extract(
        text: str,
        relation: str,
        conf_thres: float
) -> Dict[Tuple[str, str, str], float]:
    print('\tAnnotating the webpage using spacy...')
    doc = NLP(text)
    return extract_relations(
        doc, SPAN_BERT, relation,
        entities_of_interest=ENTITIES[relation],
        conf=conf_thres)


def batch_extract(
        urls: List[str],
        relation: str,
        conf_thres: float,
        extracted_tuples: dict
) -> Dict[Tuple[str, str, str], float]:
    all_res = {}
    for i in range(len(urls)):
        print(f'URL ({i+1}/{len(urls)}: {urls[i]}')
        text = get_webpage_text(urls[i])
        if not text:
            continue
        raw_res = extract(text, relation, conf_thres)
        for r in raw_res:
            _, rtype, _ = r
            if rtype == relation:
                if r in extracted_tuples:
                    pre_conf = extracted_tuples[r]
                    extracted_tuples[r] = max(pre_conf, raw_res[r])     # TODO double check this
                else:
                    extracted_tuples[r] = raw_res[r]
    return all_res


def main():
    # Retrieve user arguments
    args = get_arguments()
    # Remove punctuations from user query argument
    query = re.sub(r'[^\w\s]', '', args.q)
    # Get relation type
    relation = RELATIONS[args.r]
    # Print arguments back to user
    pprint_args(args.api_key, args.engine_id, relation, args.t, query, args.k)
    # Number of tuples to return
    k = args.k
    # List of all queries used so far
    queries = [query]

    extracted_tuples = {}
    seen_urls = set()
    iteration = 0

    while True:
        print(f'=========== Iteration: {iteration} - Query: {query} ===========')
        try:
            urls = []
            for url in search(query, args.api_key, args.engine_id, N_RETRIEVED):
                # Ignore urls that we have already processed
                if url in seen_urls:
                    continue
                urls.append(url)
                seen_urls.add(url)
            
            # Get tuples above confidence threshold
            batch_extract(urls, relation, args.t, extracted_tuples)

            # Sort extracted tuples by confidence in descending order
            extracted_tuples = {k:v for k, v in sorted(extracted_tuples.items(), key=lambda item: item[1], reverse=True)}
            # Pretty print the extracted tuples
            pprint_results(relation, extracted_tuples)
            iteration += 1
            # Check whether k tuples have been extracted
            if len(extracted_tuples) >= k:
                print(f'Total # of iterations = {iteration}')
                return
            else:
                # Generate new query term if k tuples have not yet been extracted
                for (subj, _, obj), _ in extracted_tuples.items():
                    if subj.strip().lower() not in query.lower() or obj.strip().lower() not in query.lower():
                        new_query = subj.strip() + ' ' + obj.strip()
                        if new_query not in queries:
                            queries.append(new_query)
                            query = new_query
                            break
        except KeyboardInterrupt:
            return
        except:
            print(traceback.print_exc())
            return


if __name__ == '__main__':
    main()
