import spacy
from collections import defaultdict

spacy2bert = { 
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "GPE": "LOCATION", 
        "LOC": "LOCATION",
        "DATE": "DATE"
        }

bert2spacy = {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
        }


def extract_relations(doc, spanbert, req_relation, entities_of_interest, conf=0.7):
    sentences = list(doc.sents)
    num_sentences = len(sentences)
    print("\tTotal # sentences = {}".format(num_sentences))
    res = defaultdict(int)
    n_annotation_extracted = 0
    n_overall_relations = 0
    n_extracted_relations = 0

    for i, sentence in enumerate(sentences):
        # print("\tprocessing sentence: {}".format(sentence))  # TODO comment this out before submission
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        if not (i + 1) % 5:
            print(f'\tProcessed {i + 1} / {num_sentences} sentences')

        examples = []
        for ep in entity_pairs:
            examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        if not len(examples):
            continue

        preds = spanbert.predict(examples)
        extracted_from_current_sentence = False
        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            if relation != req_relation:
                continue
            n_overall_relations += 1
            print("\n\t\t=== Extracted Relation ===")
            print("\t\tInput tokens: {}".format(ex['tokens']))
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            print("\t\tOutput Confidence: {:.3f}) ; Subject: {} ; Object: {}".format(confidence, subj, obj))
            if confidence > conf:
                if res[(subj, relation, obj)] < confidence:
                    res[(subj, relation, obj)] = confidence
                    print("\t\tAdding to set of extracted relations")
                    n_extracted_relations += 1
                    extracted_from_current_sentence = True
                else:
                    print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
            else:
                print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========")

        n_annotation_extracted += extracted_from_current_sentence

    print(f'\tExtracted annotations for  {n_annotation_extracted}  out of total  {num_sentences}  sentences')
    print(f'\tRelations extracted from this website: {n_extracted_relations} (Overall: {n_overall_relations})')
    return res


def create_entity_pairs(sents_doc, entities_of_interest: tuple, window_size=40):
    """
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    """
    entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents  # get entities for given sentence

    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower():  # make sure e1 != e2
                continue

            if 1 <= (e2.start - e1.end) <= window_size:

                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size:  # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start-gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start-gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs
