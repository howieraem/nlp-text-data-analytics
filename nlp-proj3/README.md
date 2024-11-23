This is the code repository for HW 3. 

NOTE: For this assignment, we've included a requirements.txt file to get you started.  Remember to use Python 3.6.  If you need additional libraries not included in requirements.txt, please 1) note them in your README and 2) provide your own requirements.txt file that specifies those additional libraries.

# 1 Training your own word embeddings

`1_static_embeddings` contains a minimal implementation of static word embedding evaluation via three tasks (two intrinsic and one intrinsic).  Your job is to produce word2vec and SVD-PPMI based embeddings using the Brown corpus (`data/brown.txt`) and evaluate them against these tasks.

## Evaluation
The main script to run for evaluation is `evaluate.py`.  

The three tasks used in this implementation are
* Word similarity (dataset: [WordSim353](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/)[1]; method: cosine similarity; evaluation metric: Spearman's rho)
* Analogy solving (dataset: [BATS](http://vecto.space/projects/BATS/)[2]; method: vector offset (3CosAdd); evaluation metric: accuracy)
* Paraphrase detection (dataset: [MSR Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)[3] via [this repo](https://github.com/wasiahmad/paraphrase_identification); method: logistic regression over cosine similarity; evaluation metric: accuracy)

## References
[1] [Placing search in context: The concept revisited](http://www.cs.technion.ac.il/~gabr/papers/tois_context.pdf). Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, Ehud Rivlin, Zach Solan, Gadi Wolfman, and Eytan Ruppin. ACM Transactions on information systems, Vol. 20, No. 1, pp. 116-131. January 2002.

[2] [Analogy-based detection of morphological and semantic relations with word embeddings: what works and what doesn't](https://www.aclweb.org/anthology/N16-2002/). In Proceedings of the NAACL Student Research Workshop, pages 8–15. June 2016.

[3] [Unsupervised construction of large paraphrase corpora: Exploiting massively parallel news sources](https://www.aclweb.org/anthology/C04-1051/). COLING 2004. August 2004.

# 2 Contextualized word embeddings: BERT

`2_bert_wic` contains starter code for training a BERT-based classifier on recognizing polysemous words based on their context.  Your job is to complete the code, extracting BERT-based representations for each example and training a classifier using those representations. 

`bert_wic.py` contains 1) fixed command line arguments 2) methods to parse the [Word-in-Context (WIC)](https://pilehvar.github.io/wic/) corpus, included in the `wic` directory.  Do not change its arguments. An accuracy of 0.6 will receive full credit, provided the model trains in under 10 minutes.

# 3 Probing BERT

`3_explore_bert` contains pickled hidden state BERT representations from every layer of `bert-base-cased` for three evaluation corpora: the CONLL 2003 task, which contains sentences annotated with 1) part-of-speech and 2) named entity tags and SemEval 2010 Task 8 which contains sentences annotated with 3) entities and the relation expressed between those entities by the sentence.

We've pre-processed these tasks for you to make them easier to work with.  Use `torch.load(*.pt)` to load the pre-processed input file.

`conll.pt` contains 2 keys, `train` and `validation`.  Each key maps to an array where each element is a dictionary corresponding to an example in CONLL.  Its keys are:
* `sentence` - the original sentence in CONLL
* `pos_labels` - the part-of-speech labels for the words in `sentence`
* `ner_labels` - the named entity labels for the words in `sentence`
* `word_token_indices` - the indices of the BERT representations for each words in `sentence` (because BERT uses wordpiece tokenization, each word may map to multiple tokens)
* `hidden_states` - a 13-tuple where each element i is a [sequence length x 768] tensor for the BERT representations drawn from the i-th layer; note that the sequence length may be longer than the number of words in the sentence (the lengths of `pos_labels` / `ner_labels`) due to wordpiece tokenization.  You will need to use `word_token_indices` to recover the and mean-pool the right hidden states.

`semeval.pt` contains 2 keys, `train` and `validation`. Each key maps to an array where each element is a dictionary corresponding to an example in SemEval.  Its keys are:
* `sentence` - the original sentence in SemEval
* `rel_label` - the relation label for the entities marked in `sentence`
* `entity1_representations` - a 13-tuple where each element i is a 768-dimension tensor produced by mean pooling the tokens corresponding to entity 1 in `sentence` from the i-th layer of BERT
* `entity2_representations` - a 13-tuple where each element i is a 768-dimension tensor produced by mean pooling the tokens corresponding to entity 2 in `sentence` from the i-th layer of BERT

For every layer's representations, train logistic regression classifiers on 1) POS tagging (using CONLL), 2) named entity recognition (using CONLL) and 3) relation extraction (using SemEval).  Plot the performance of your classifiers on each task's validation set in a single graph: macro averaged F1 on the Y axis, the layer number on the X axis with a separate line for each task. 

NOTE: For the POS tasks, some labels from the validation set may not exist in train set due to the sampling we applied -- please ignore them.

# Submission Instructions

Please submit a single zipped directory (named {YOUR_UNI}\_hw3.zip) on Courseworks containing:
1 - `train_embeddings.py`, the code you use to train your word word2vec and PPMI-SVD based embeddings on the Brown corpus
2 - `svd_embeddings.txt`, your best performing PPMI-SVD based embeddings
3 - `bert_wic.py`, your code to train and evaluate a BERT-based word-in-context (WiC) classifier
4 - `explore_bert.py`, your code that probes BERT's representations, producing a plot of the macro-F1 of each hidden layer on part-of-speech tagging, named entity recognition and relation classification