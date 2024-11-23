# Proj 1

## Environment Configuration
Python version tested is 3.6.x. Creating a dedicated environment via `conda` or `virtualenv` is recommended.
Once the environment is activated, please install the following packages via `pip3` or `conda`:
```
numpy
scipy
scikit-learn
pandas
json
nltk
```


## Run

Activate the environment and then run the following command:
```
python3 hw1.py --train <path_to_train_jsonl> \
    --test <path_to_test_jsonl> \
    --user_data <path_to_user_data_json> \
    --model <model_type> \
    --lexicon_path <path_to_lexicon_directory> \
    --outfile <path_to_output_file>
```

This will train and test the models at the same time. Please see `Model` in `hw1.py` for more details.
Note that it might take a few minutes and print a few things to the console:
- `nltk` punkt data download. Please ensure the test machine has either been connected to the Internet or already downloaded the required package.
- `CovergenceWarning` from scikit-learn: max_iter was reduced to avoid overfitting, please ignore.
- Religious-only, non-religious and overall accuracies.


## Features

### Model 1: For religious data

1. Ngram: n = 1, 2, 3
2. Lex: Comparison between debaters' connotation word counts. 1 means Pro > Con, 0 otherwise. It was found that performance was better without positive word count comparison.
3. Ling:  
    i. Comparison between debaters' text lengths (the numbers of tokens). 1 means Pro > Con, 0 otherwise.   
    ii. Comparison between debaters' pronoun counts. 1 means Pro > Con, 0 otherwise. It was found that the model was better without 2nd person pronoun counts.
4. User:  
    i. Comparison between debaters' big issue opinions. Opinions were firstly transformed to values: Pro was 0, Con was 1 and others 0.5. Then, feature value 1 means opinion values equal, 0 otherwise. 
    ii. Comparison between debaters' political ideologies. 1 means Pro == Con, 0 otherwise.

### Model 2: For non-religious data

1. Ngram: (same as above)
2. Lex: Average V, A and D values over tokens in each debate for both Pro and Con sides, so it is 6-dimensional per debate.
3. Ling: (same as above)
4. User:  
    i. (same as above) 
    ii. Friendship feature: 1 means the debaters are on each other's friends list, 0 otherwise.

### Limitations

- Model 1 Ngram+Lex performed better than model 2 Ngram+Lex in religious debates but worse than model 2 Ngram+Lex in non-religious debates.
However, both with length and pronoun linguistic features incorporated and without user features, model 1 performed better in non-religious debates and model 2
performed better in religious debates.

- In the current implementation, a valid test *.jsonl file must be given. Feature extractions, model training and model predictions 
on the entire training and test data happened at the same time, though prediction results could be retrieved later with indices
(e.g. those of religious debates). In the future, selective feature extraction/prediction or singleton test sample inference 
should be implemented.


## Evaluation

Quick note: Ngram+VAD not working well with bigIssues+politic features, regardless of what Ling features.

| Model | Religious | Non-religious | Overall |
| --- | --- | --- | --- |
| Ngram | 0.6559 | 0.8039 | 0.7694 |
| Ngram+VAD (1) | 0.6559 | 0.8072 | 0.7719 |
| Ngram+Conno123 | 0.6559 | 0.7974 | 0.7644 |
| Ngram+Conno12 | 0.6452 | 0.7941 | 0.7594 |
| Ngram+Conno13 | 0.6344 | 0.7908 | 0.7544 |
| Ngram+Conno23 (2) | 0.6882 | 0.7908 | 0.7669 |
| Ngram+VAD+length+modalVerb | 0.6774 | 0.8007 | 0.7719 |
| Ngram+VAD+length+pronoun (3) | 0.7097 | 0.7908 | 0.7719 |
| Ngram+VAD+modalVerb+pronoun | 0.6774 | 0.8007 | 0.7719 |
| Ngram+Conno23+length+modalVerb | 0.6667 | 0.7941 | 0.7644 |
| Ngram+Conno23+length+pronoun (4) | 0.6882 | 0.8007 | 0.7744 |
| Ngram+Conno23+modalVerb+pronoun | 0.6774 | 0.7810 | 0.7569 |
| Ngram+VAD+length+pronoun+bigIssues+friend (5) | 0.6774 | 0.8007 | 0.7719 |
| Ngram+VAD+length+pronoun+bigIssues+politic | 0.6882 | 0.7908 | 0.7669 |
| Ngram+VAD+length+pronoun+friend+politic | 0.7097 | 0.7941 | 0.7744 |
| Ngram+Conno23+length+pronoun+bigIssues+friend | 0.6989 | 0.7908 | 0.7694 |
| Ngram+Conno23+length+pronoun+bigIssues+politic (6) | 0.7312 | 0.7908 | 0.7769 |
| Ngram+Conno23+length+pronoun+friend+politic | 0.6882 | 0.8007 | 0.7744 |
| Ngram+Lex best (2) + (1) | 0.6882 | 0.8072 | 0.7794 |
| Ngram+Lex+Ling best (4) + (3) | 0.6882 | 0.7908 | 0.7669 |
| Ngram+Lex+Ling best (3) + (4) Swapped | 0.7097 | 0.8007 | 0.7794 |
| Ngram+Lex+Ling+User best (6) + (5) | 0.7312 | 0.8007 | 0.7845 |


## Feature Data Analysis

### Lexicon Features

#### Connotation Comparison
Compare connotation token counts between debaters for each doc. 1 means Pro > Con, 0 otherwise. 
The Pro means and Con means of negative and neutral comparisons were further apart than those of positive comparisons.
Statistics below might indicate that in the winner's document, the number of negative and neutral tokens would be probably larger.

| Cls | Mean (pos, neg, neu) | Std (pos, neg, neu) |
| --- | --- | --- |
| Pro | (0.6755, 0.6915, 0.6862) | (0.4682, 0.4619, 0.4640) |
| Con | (0.2559, 0.2512, 0.2322) | (0.4364, 0.4337, 0.4223) |


#### VAD Average Values
VAD average values over tokens, for pro side and con side of each doc, and compare between debaters

| Cls | Mean Pro (V, A, D) | Std Pro (V, A, D) | Mean Con (V, A, D) | Std Con (V, A, D) |
| --- | --- | --- | --- | --- |
| Pro | (0.5821, 0.4606, 0.5659) | (0.0391, 0.0334, 0.0317) | (0.5697, 0.4617, 0.5497) | (0.0845, 0.0517, 0.0701) |
| Con | (0.5835, 0.4618, 0.5604) | (0.0563, 0.0353, 0.0392) | (0.5807, 0.4587, 0.5659) | (0.0590, 0.0422, 0.0482) |


### Linguistic Features

#### Length Compare
Compare the number of tokens between debaters per doc. 1 means Pro > Con, 0 otherwise.

| Cls | Mean | Std |
| --- | --- | --- |
| Pro | 0.6968 | 0.4596 |
| Con | 0.2275 | 0.4192 |


#### Modal Verb Compare
Compare the number of modal verbs between debaters per doc. 1 means Pro > Con, 0 otherwise.

| Cls | Mean | Std |
| --- | --- | --- |
| Pro | 0.6702 | 0.4701 |
| Con | 0.2654 | 0.4415 |


#### Pronoun Compare
Compare the number of 1st/2nd/3rd person pronouns between debaters per doc. 1 means Pro > Con, 0 otherwise.

| Cls | Mean (1, 2, 3) | Std (1, 2, 3) |
| --- | --- | --- |
| Pro | (0.6170, 0.5213, 0.6702) | (0.4861, 0.4995, 0.4701) |
| Con | (0.2844, 0.3602, 0.2986) | (0.4511, 0.4801, 0.4576) |


### User Features

#### Big Issues Ideology
1 means the debaters have the same opinion on the same issue, 0 otherwise.

| Cls | Mean | Std |
| --- | --- | --- |
| Pro | 0.0904 | 0.2868 |
| Con | 0.0758 | 0.2647 |


#### Political Ideology
1 means the debaters have the same political ideology, 0 otherwise.

| Cls | Mean | Std |
| --- | --- | --- |
| Pro | 0.0904 | 0.2868 |
| Con | 0.0758 | 0.2647 |


#### Friend
1 means the debaters are BOTH on each other's friend list, 0 otherwise.

| Cls | Mean | Std |
| --- | --- | --- |
| Pro | 0.2287 | 0.4200 |
| Con | 0.1517 | 0.3587 |
