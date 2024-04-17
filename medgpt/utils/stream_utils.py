from collections import defaultdict
from medgpt.datasets.utils import get_all_splits
import logging
import os
import pickle
import re

def get_entities_for_doc(docs, doc_id):
    r''' Return entities for the given doc_id from the docs dictionary.

    docs:
        Output of medcat multiprocessing
    doc_id:
        id of the doc in docs
    '''
    ents = docs[doc_id]['entities']
    # Depending on the version of medcat ents can be dict {id: entities, ...} or list of entities
    ents = ents.values() if isinstance(ents, dict) else ents

    return ents

'''
def get_patient_count_per_token(save_path, dataset, force=False):
    if not force and save_path is not None and os.path.exists(save_path):
        print("Loading an existing version, remove if you want to rebuild.")
        print(save_path)
        token_cnt = pickle.load(open(save_path, 'rb'))
    else:
        token_cnt = defaultdict(int)
        for _dataset in get_all_splits(dataset):
            for stream in _dataset['stream']:
                unique_tokens = set([sample['token'] for sample in stream])
                for token in unique_tokens:
                    token_cnt[token] += 1
        token_cnt = dict(token_cnt)

        if save_path:
            print("Saving to: ", save_path)
            pickle.dump(token_cnt, open(save_path, 'wb'))

    return token_cnt
'''

def get_token_counts_from_dataset(doc_paths, doc2info, meta_requirements=None, save_path=None, force=False):
    if not force and (save_path is not None and os.path.exists(save_path)):
        print("Loading an existing version, remove if you want to rebuild.")
        print(save_path)
        pt2cui2cnt = pickle.load(open(save_path, 'rb'))
    else:
        pt2cui2cnt = defaultdict(lambda: defaultdict(int))
        for path in doc_paths:
            print("Loading: ", path)
            docs = pickle.load(open(path, 'rb'))

            # Frequency for each each entity given a patient
            for doc in docs:
                for ent in get_entities_for_doc(docs, doc):
                    # Must match all meta meta_anns
                    if not meta_requirements or \
                       all([ent['meta_anns'][name]['value'] == value for name, value in meta_requirements.items()]):
                        cui = ent['cui']
                        pt = doc2info[doc]['pt']
                        pt2cui2cnt[pt][cui] += 1

        pt2cui2cnt = dict(pt2cui2cnt)
        if save_path:
            print("Saving to: ", save_path)
            pickle.dump(pt2cui2cnt, open(save_path, 'wb'))
    return pt2cui2cnt


def docs2stream(doc_paths, doc2info, pt2cui2cnt, doc2text=None, meta_requirements={}, entity_type_column='tuis',
                historical_meta=None, historical_meta_value=None, old_pt2stream=None, skip_cuis=None,
                require_time=True, save_path=None, tokenizer=None, cntx_size=30, force=False, sentence_limits=None):
    r''' Convert the `docs` output of medcat multiprocessing
    to a stream of concepts for each patient.

    Args:
        docs
        meta_requirements:
            Values for meta_annotaitons that must exist e.g. = {'Presence': True}
        historical_meta:
            Do not use if cntx_size > 0
    '''

    if cntx_size > 0 and historical_meta is not None:
        raise Warning("Using both historical meta and cntx_size, is bound to screw things up")

    if not force and save_path is not None and os.path.exists(save_path):
        print("Loading an existing version, remove if you want to rebuild.")
        print(save_path)
        pt2stream = pickle.load(open(save_path, 'rb'))
    else:
        pt2stream = defaultdict(list)
        have_warned = set()

        for path in doc_paths:
            print("Loading: ", path)
            docs = pickle.load(open(path, 'rb'))

            for doc in docs:
                tokens = None
                if doc2text is not None and tokenizer is not None:
                    tokens = tokenizer(doc2text[doc])
                    # Add a space to a token if it does not have one
                prev_tkn_ind = 0

                for ent in get_entities_for_doc(docs, doc):
                    if not meta_requirements or \
                       all([ent['meta_anns'][name]['value'] == value for name, value in meta_requirements.items()]):

                        cui = ent['cui']

                        if skip_cuis is None or cui not in skip_cuis:
                            if doc2info is not None and doc in doc2info and 'time' in doc2info[doc]:
                                timestamp = doc2info[doc]['time'].timestamp()
                            elif 'document_timestamp' in ent:
                                timestamp = ent['document_timestamp']
                            else:
                                timestamp = None # Means time is not known, later it will be ignored if necessary

                            if not require_time or timestamp is not None: # Skip all where timestamp is None
                                if historical_meta is not None and timestamp is not None:
                                    # If something is historical then make the timestamp less by 1 because it appeared before 
                                    #other things in this document. Unles time is None which means time is undefined
                                    if ent['meta_anns'][historical_meta]['value'] == historical_meta_value:
                                        timestamp = timestamp - 1

                                pt = doc2info[doc]['pt']
                                cnt = pt2cui2cnt[pt][cui]
                                cntx_left = None
                                cntx_right = None
                                if ent[entity_type_column]: # This can be none in some cases
                                    token_type = ent[entity_type_column][0]
                                else:
                                    token_type = 'unk'
                                    if cui not in have_warned:
                                        logging.warning(f"Entity type missing from: {cui}")
                                        have_warned.add(cui)
                                if doc2text is not None and tokenizer is not None:
                                    # Find start and and position
                                    tkn_start = -1
                                    tkn_end = -1
                                    for _tkn_ind, tkn in enumerate(tokens[prev_tkn_ind:]):
                                        tkn_ind = prev_tkn_ind + _tkn_ind
                                        # We want to take a word if its start (or end) is inside of ent bounds,
                                        #that means we take all tokens that with any portion are inside of the entity.
                                        if (tkn[1][0] >= ent['start'] and tkn[1][0] <= ent['end']) or \
                                           (tkn[1][1] >= ent['start'] and tkn[1][1] <= ent['end']) or \
                                           (ent['start'] >= tkn[1][0] and ent['end'] <= tkn[1][1]):
                                            if tkn_start == -1:
                                                tkn_start = tkn_ind
                                            #always set end
                                            tkn_end = tkn_ind
                                        if tkn[1][1] >= ent['end']:
                                            prev_tkn_ind = max(0, _tkn_ind - 1)
                                            break
                                    if sentence_limits and cntx_size:
                                        # We want to find the 
                                        for i in range(0, cntx_size+1):
                                            if (tkn_start - i) > 0:
                                                end_word = tokens[tkn_start - i - 1][0]
                                                start_word = tokens[tkn_start - i][0]
                                                if (end_word and end_word.endswith(sentence_limits)) and (start_word.strip() and start_word.strip()[0].isupper()):
                                                    # Means this token is the beginning of the sentence wehre the entity is found
                                                    _cntx_left_start = tkn_start - i
                                                    break
                                                elif i == cntx_size:
                                                    # Take the max context as no sentence start was found
                                                    _cntx_left_start = tkn_start - i
                                            else:
                                                _cntx_left_start = 0
                                                break

                                        for i in range(0, cntx_size + 1):
                                            if (tkn_start + i + 1) < len(tokens):
                                                end_word = tokens[tkn_start + i][0]
                                                start_word = tokens[tkn_start + i + 1][0]
                                                if (end_word and end_word.endswith(sentence_limits)) and (start_word.strip() and start_word.strip()[0].isupper()):
                                                    _cntx_right_end = tkn_start + i + 1
                                                    break
                                                elif i == cntx_size:
                                                    _cntx_right_end = tkn_start + i
                                            else:
                                                _cntx_right_end = len(tokens)
                                                break
                                    elif cntx_size:
                                        _cntx_left_start = max(0, tkn_start - cntx_size)
                                        _cntx_right_end = min(len(tokens), tkn_end + cntx_size)
                                    else:
                                        raise Exception("cntx_size or contx_size_sentence - have to be set")

                                    cntx_left =  [(re.sub("([^0-9])\\1{3,}", "\\1\\1\\1", token[0]).replace("[**", " ").replace("**]", ' '),
                                                   _i + _cntx_left_start)
                                                 for _i, token in enumerate(tokens[_cntx_left_start:tkn_start])]
                                    cntx_right = [(re.sub("([^0-9])\\1{3,}", "\\1\\1\\1", token[0]).replace("[**", " ").replace("**]", ' '),
                                                   _i + tkn_end + 1)
                                                 for _i, token in  enumerate(tokens[tkn_end+1:_cntx_right_end])]
                                    assert len(cntx_left) <= cntx_size and len(cntx_right) <= cntx_size, \
                                           f'Cntx {len(cntx_left)} {len(cntx_right)} too big for: {str(doc)}'
                                    """
                                    if len(cntx_left) > cntx_size or len(cntx_right) > cntx_size:
                                        print(tkn_start, tkn_end, _cntx_left_start, _cntx_right_end, cui, doc)
                                        print(ent['start'], ent['end'])
                                        print(prev_tkn_ind)
                                        print()
                                    """
                                token = ' ' + str(cui) + ' ' #Just in case cui gets spaces from both sides
                                pt2stream[pt].append((token, cui, cnt, timestamp, token_type, doc,
                                                      [cl[0] for cl in cntx_left], [cl[1] for cl in cntx_left],
                                                      [cr[0] for cr in cntx_right], [cr[1] for cr in cntx_right],
                                                      ent['meta_anns']['Presence']['value'],
                                                      tkn_start)) # tkn_start is the position of the entity itself in the split text
        pt2stream = dict(pt2stream)
        if save_path:
            print("Saving to: ", save_path)
            pickle.dump(pt2stream, open(save_path, 'wb'))

    return pt2stream


def get_patient_count_per_token(pt2stream, force=False, save_path=None):
    if not force and save_path is not None and os.path.exists(save_path):
        print("Loading an existing version, remove if you want to rebuild.")
        print(save_path)
        token_cnt = pickle.load(open(save_path, 'rb'))
    else:
        # Calculate counts
        token_cnt = defaultdict(int)

        for pt, v in pt2stream.items():
            all_cuis = set([x[0] for x in v])
            for cui in all_cuis:
                token_cnt[cui] += 1
        token_cnt = dict(token_cnt)

        if save_path:
            print("Saving to: ", save_path)
            pickle.dump(token_cnt, open(save_path, 'wb'))

    return token_cnt
