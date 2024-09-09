# Project 2

## Team Members
- J Lin
- F Cao


## Files
- `pytorch_pretrained_bert/`: From spaCy-SpanBERT repo
    - `__init__.py`
    - `fine_utils.py`
    - `modeling.py`
    - `tokenization.py`
- `download_finetuned.sh`: Shell script to download the fine-tuned SpanBERT model
- `main.py`: Where it's at!
- `relations.txt`: Relations list from spaCy-SpanBERT repo
- `requirements.txt`: Packages to install
- `spacy_help_functions.py`: Uses spacy to extract relations
- `spanbert.py`: From spaCy-SpanBERT repo


## How to Configure Environment and Run
- Create a new virtual environment https://docs.python.org/3/tutorial/venv.html
- Activate the environment you created
```
source {env_name}/bin/activate
```
- `cd` to the code directory
- Install/update dependencies
```
pip3 install -U pip setuptools wheel
pip3 install -r requirements.txt
```
- Download the fine-tuned SpanBERT model, this should create a pretrained_spanbert folder in the current directory with two files, `config.json` and `pytorch_model.bin`
```
bash download_finetuned.sh
```
- Run the iterative set expansion script with the following command
```
python3 main.py {google_api_key} {google_engine_id} {r} {t} {q} {k}

- google_api_key: Provided in the last section below
- google_engine_id: Provided in the last section below
- r: Integer between 1 and 4, indicating the relation to extract:
    - 1 is for Schools_Attended
    - 2 is for Work_For
    - 3 is for Live_In
    - 4 is for Top_Member_Employees
- t: Real number between 0 and 1 indicating the extraction confidence threshold
- q: A seed query that should be enclosed in double quotes
- k: an integer greater than 0, indicating the number of tuples that we request in the output

You can also find more help by running python3 main.py --help
```


## Internal Design
The design mainly consists of the following components:

- `pytorch_pretrained_bert/*`: contains definitions of the underlying BERT-like neural network models, NLP tokenizers (convert text to tokens/words), and some utilities for loading/downloading files
- `spanbert.py`: defines the SpanBERT model and its associated inference code (incl. input preprocessing, prediction, and conversion from indices to labels)
- `spacy_help_functions.py`: utilizes `spacy` to process sentences and named entities, and then feeds these into SpanBERT to extract relations
- `main.py`: where the main loop is. It:
  1. Parses user arguments (see `get_arguments()` and the first few lines of `main()`)
  2. Obtains URLs from the Google search results of a query (see `search()`)
  3. Scrapes text from raw HTML for each URL (see `batch_extract()` and `get_webpage_text()`)
  4. spaCy processes the text and SpanBERT extracts relations specified, by calling `spacy_help_functions.py` (see `extract()`)
  5. Relations of these URLs are aggregated (see the last few lines of `batch_extract()`)
  6. The loop in `main()` determines whether to change the query and start a new iteration (search -> scrape -> process text -> extract relations -> aggregate). We update the query by using the tuple with the highest confidence level, but only if it was not used before, and the terms don't already exist in the last query term used.

External libraries used:

- `google-api-python-client`: returns a list of URLs given a query
- `requests`: loads raw HTML contents given a URL
- `beautifulsoup4`: extracts text from raw HTML contents
- spaCy: converts text to some sentences, and then extracts subject entities and object entities from these sentences
- SpanBERT and relevant libraries (such as PyTorch): the deep learning model which takes named entities and tokens as inputs and then outputs relations

## Detailed Description of Processing
  1. We retrieve the webpage using the requests library within the function `get_webpage_text`, which has parameters `url`, `timeout` that we set to 20 secs, and `max_chars` that is set to 20,000. We attempt to get the contents of the url, and if the time exceeds 20 seconds, we skip it and return an empty string.
  2. We extract the plain text from the retrieved url by using the BeautifulSoup library, also within the same function mentioned above. We parse out the text inside using the html.parser feature. We then perform additional cleaning and retrieve only text nested in \<p\> and \<li\>tags, and only if the contents inside the tags are text, thus removing characters such as \\n and \\t. From those two lists, we combine them into one by iterating over each list item in sequential order. Finally, we join these paragraphs together to create the final string and store it in `text`.
  3. If the `text` returned in the previous step surpasses 20,000 characters, then we trim it and only return the first 20,000 characters. This happens in the same `get_webpage_text` function.
  4. We use the spaCy library to split the text returned from the previous step to sentences, tokenize the terms, and extract all named entities such as 'PERSON' or 'ORGANIZATION'. This is completed by loading the spaCy model `en_core_web_lg`. We then apply that model to our text. This is completed within the `extract` function which takes three parameters: `text` which is the output from step 3, `relation` which is 1 of the 4 relations the user can pick from, and `conf_thres` which is the minimum threshold for extracted relations that we are willing to keep.
  5. Within the same `extract` function, we run the `extract_relations` function from `spacy_help_functions.py` to extract all instances of the relation as specified by our parameter `relation`. We use SpanBERT to predict the corresponding relations, and we store the extracted tuples in a dictionary, with valid tuples being those above the confidence threshold. We also ignore any duplicates that have a lower confidence than an existing record.
