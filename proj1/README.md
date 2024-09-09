# Project 1

## Team Members
- F Cao
- J Lin

## Submitted Files
- query.py: Where it's at!
- requirements.txt: Stores packages to install
- stopwords.txt: List of stopwords

## How to Run
- Create a new virtual environment https://docs.python.org/3/tutorial/venv.html
- Activate the environment you created `source {env_name}/bin/activate`
- `cd` to the code directory
- Install dependencies `pip install -r requirements.txt`
- Run `python3 query.py {google_api_key} {google_search_engine_id} {precision} {query}`, where `{query}` should be enclosed in double quotes

## Internal Design Description
We use two external libraries in our project, googleapiclient and numpy. Up top, after our import statements, we set a few variables that are used later on. `ALPHA`, `BETA`, `GAMMA` values are used when we run Rocchio's algorithm for the query expansion. `N_RETRIEVED` sets the number of documents Google will pull in each iteration. Lastly, `STOP_WORDS` is the set of all words in stopwords.txt document.

Our program starts off by getting the user arguments, which in our case would be the Google API key, the Google Search Engine ID, the precision, and the query terms. We process the query terms to remove punctuation, and we then start our while loop.

We first format the arguments the user entered and return it on the terminal. We then search using Google Search Engine for the query term with the `search` function. In the first iteration, we test whether the number of documents returned is less than 10, and if it is, terminate the program. If 10 documents are returned, we proceed to return the search results to the user. We iterate through the array of results, and use the `pprint_result` function to format it in the terminal. While we display each query result to the user, we also ask the user about its relevancce. We use a while loop to guarantee the input is valid (y/n). Also during this process, we append the snippet as an array from each result to the documents array.

After each iteration of presenting the query results, we check whether the precision has reached the target. We calculate the precision using the `calc_precision` function, and format the feedback for the user using the `feedback` function. The feedback function returns True if we achieved our desired precision, or if the current precision is equal to 0, in which case it should also terminate. Otherwise, it returns False and our while loop continues.

The next step in this process is the query expansion steps. We first run the `preprocessing` function on the documents array. The `preprocessing` function converts all words to lower case, removes words that are non-english characters, and words that are found in `STOP_WORDS` set. We then run the function `get_vectors` on the returned documents from the preprocessing step and the last query searched to convert the query and documents to vectors.

Following that, we run the `refine_vector` function on the returned vectors according to the user's feedback, and run the `rocchio` function to run Rocchio's algorithm and produce a new vector. Lastly, we run the `expand_query` function, which we will talk about in more detail in the next section. The while loop will break either when we reached the desired precision, have a precision of 0, or if there is a keyboard interruption.

## Query-Modification Method
We first transform the query string and the preprocessed document words into vectors. In the `get_vectors` function, the keys of `term_docs` are the document words (i.e. the vocabulary) and the values are the indices of documents where the words appear. This dictionary is then used to obtain the document frequency for each document word. We then derive a `word2idx` dictionary from the vocabulary to map words to word indices. Next, we calculate the tf-idf score for all words in the query and all the documents respectively. As for the term frequency (tf) weight, we find that the log normalization variant works better. The inverse document frequency smooth variant is selected as the inverse document frequency (idf) weight. We assign tf*idf value of a word in a document (or a query) to the corresponding word index in that document's (or query's) vector.

Once vectors are produced, we perform Rocchio's algorithm to obtain a new vector for selecting new words in the next stage. In the `refine_vector` function, the document vectors are divided into two groups based on whether relevant/irrelevant marked by the user. Suppose we have three factors alpha, beta and gamma. Rocchio's algorithm outputs a new vector which equals alpha * query vector + beta * average of relevant vectors - gamma * average of irrelevant vectors. Our `alpha` value is set to 1.0, `beta` to 0.8, and `gamma` to 0.1.

We select the new words with the top corresponding values in the vector from Rocchio's algorithm, as done in the beginning of `expand_query`. Given the indices of the top words, we pick two new words from the vocabulary which are not in the existing query.

With the new words, we then rearrange the ordering of this set based on our vector results. We store a temporary array called `test` that converts the set of all words to lowercase, for easier comparison with our `value` vector. We then store in a dictionary `tmp` the key value pairs with the keys being the term, and the value being the weight. Becaues the `tmp` dictionary keys are all lowercased, and we would like to preserve casing, we create another dictionary `final_dict` which will store query words based on their original casing. `final_dict` is then sorted based on the values in descending order, and we return the reordered query as a string.
