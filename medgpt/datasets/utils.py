import numpy as np
import random
from datetime import datetime
import math
import datasets
import random
import copy

def get_all_splits(dataset):
    all_datasets = []
    if 'train' in dataset:
        all_datasets.append(dataset['train'])
    if 'test' in dataset:
        all_datasets.append(dataset['test'])
    if 'valid' in dataset:
        all_datasets.append(dataset['valid'])
    if isinstance(dataset, datasets.arrow_dataset.Dataset):
        # If we have only one, ie no train/test
        all_datasets.append(dataset)

    return all_datasets

def make_example(token, ent_example, token_type='unk', cnt=10**6, time=None, cntx=None, prefix=''):
    # TODO - fix cntx, it does not do anything
    out = {'token': prefix + token,
           'cui': '',
           'cnt': cnt,
           'time': time,
           'token_type': token_type,
           'doc_id': ent_example['doc_id'],
           'cntx_left': [],
           'cntx_left_inds': [],
           'cntx_right': [],
           'cntx_right_inds': [],
           'presence': None,
           'ent_tkn_id': None,
           }
    return out


def get_duration_separator(start_time, current_time, prefix=''):
    'Times are approximated, not really important'
    days = (current_time - start_time) // (24*60*60) + 1
    if days < 7:
        sep = prefix + '{} days later'.format(days)
    elif days < 30:
        weeks = days // 7
        sep = prefix + '{} weeks later'.format(weeks)
    elif days < 365:
        months = days // 30
        sep = prefix + '{} months later'.format(months)
    else:
        years = days // 365
        sep = prefix + '{} years later'.format(years)

    return sep


def bucket_concepts(examples, bucket_size_seconds=365*24*60*60, separator_format='after {} days', time_prefix=''):
    r''' Will bucket concepts into specified bucket_size.

    Args:
        examples
    '''
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]

        new_stream = []
        _bucket = []
        _tokens = set()
        start_time = -1
        prev_ent = None
        for ent in stream:
            if start_time == -1:
                start_time = ent['time']

            if ent['time'] - start_time >= bucket_size_seconds:
                # Add to stream
                new_stream.extend(_bucket)
                _bucket = []
                _tokens = set()

                # This will have different separator for different time spans
                _separator = get_duration_separator(prev_ent['time'], ent['time'], prefix=time_prefix)
                # A separator is +1 of the last token in the stream
                new_stream.append(make_example(ent_example=ent, token=_separator, token_type='time_sep', cnt=10**6, time=new_stream[-1]['time']+1))
                # Change start time to current entity time
                start_time = ent['time']

            if ent['token'] not in _tokens:
                _bucket.append(ent)
                _tokens.add(ent['token'])

            prev_ent = ent

        if _bucket:
            new_stream.extend(_bucket)

        examples['stream'][i] = new_stream
        new_stream = []

    return examples

def add_position_ids(examples, separators=set()):
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]

        old_t = None
        cnt = 0
        for ent in stream:
            ent['position_ids'] = cnt
            if ent['token'] in separators:
                cnt += 1

    return examples

def create_tte_prediction_timelines(examples, prefixes, token_type, n_tte=1):
    r'''
        n_risk: If there are multiple prefixes one example can be used to generated multiple risk prediction timelines, this determines how many.
        min_time: If set, at least that amount of time has to pass from middle of the timeline until the event
    '''
    new_examples = {'patient_id': [], 'stream': []}

    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        pt_id = examples['patient_id'][i]

        time_sep = np.median([x['time'] for x in stream])
        _prefixes = prefixes
        for _ in n_tte:
            new_stream = []
            tte_concepts = set()
            old_concepts = set()
            one_new = None
            for ent in stream:
                if ent['time'] < time_sep:
                    new_stream.append(ent)
                    old_concepts.add(ent['cui'])
                elif ent['token_type'] == token_type:
                    if one_new is None:
                        one_new = ent
                        
                        # Random select an element from the _prefixes array and remove it from _prefixes
                        el = random.choice(_prefixes)
                        _prefixes.remove(el)
                        timediff, prefix = el
                    if ent['time'] <= time_sep + timediff:
                        if ent['cui'] not in old_concepts:
                            risk_concepts.add(ent['cui'])
                    else:
                        # We are over the timelimit
                        break

            ent = stream[-1] 
            if one_new is not None and risk_concepts and len(risk_concepts) > 0:
                one_new['token'] = one_new['cui'] = " ".join(risk_concepts)
                one_new['cntx_left'] = prefix.split(" ")
                one_new['cntx_left_inds'] = list(range(len(one_new['cntx_left'])))
                one_new['doc_id'] = ent['doc_id'] # Same as the last token
                one_new['cntx_right'] = []
                one_new['cntx_right_inds'] = []
                one_new['presence'] = None
                one_new['token_type'] = f'risk-{timediff}-{token_type}'
            
                new_stream.append(one_new)
                new_stream.append(ent) # Append the </s> token, ie last token
                #examples['stream'][i] = new_stream
                new_examples['stream'].append(new_stream)
                new_examples['patient_id'].append(pt_id)
 
    return new_examples

def create_risk_prediction_timelines_but_better(examples, prefixes, token_type, n_risk=1, min_past_length=1, max_timeline_len=None, min_risk_concepts=5):
    r'''
        max_timeline_len is in number of concepts, not seq_len
        min_past_length - also number of concepts
        n_risk: If there are multiple prefixes one example can be used to generated multiple risk prediction timelines, this determines how many.
    '''
    new_examples = {'patient_id': [], 'stream': []}

    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        pt_id = examples['patient_id'][i]

        middle = len(stream) // 2
        current_time = None
        _prefixes = list(prefixes) # Copy
        for _ in range(n_risk):
            new_stream = []
            risk_concepts = []
            old_conf_cuis = set()
            one_new = None
            over_time_limit = False

            # Add until maximum concepts
            for ent in stream:
                if len(new_stream) < min(max(middle, min_past_length), max_timeline_len):
                    new_stream.append(ent)
                    current_time = ent['time']
                    if ent['token_type'] == token_type: # To avoid hypo and negated
                        old_conf_cuis.add(ent['cui'])
                elif ent['token_type'] == token_type:
                    if one_new is None:
                        one_new = ent
                        
                        # Random select an element from the _prefixes array and remove it from _prefixes
                        el = random.choice(_prefixes)
                        _prefixes.remove(el)
                        timediff, prefix = el
                    if ent['time'] <= current_time + timediff:
                        if ent['cui'] not in old_conf_cuis and ent['cui'] not in risk_concepts:
                            risk_concepts.append(ent['cui'])
                    else:
                        # We are over the timelimit, this is a requirement, otherwise timeline is shorter than the timelimit
                        over_time_limit = True
                        break

            ent = stream[-1] 
            if one_new is not None and len(risk_concepts) >= min_risk_concepts and over_time_limit:
                one_new['token'] = one_new['cui'] = " ".join(risk_concepts)
                one_new['cntx_left'] = prefix.split(" ")
                one_new['cntx_left_inds'] = list(range(len(one_new['cntx_left'])))
                one_new['doc_id'] = ent['doc_id'] # Same as the last token
                one_new['cntx_right'] = []
                one_new['cntx_right_inds'] = []
                one_new['presence'] = None
                one_new['time'] = current_time + 10 # Just move the current time a bit
                one_new['token_type'] = f'risk-{timediff}-{token_type}'
            
                new_stream.append(one_new)
                new_stream.append(ent) # Append the </s> token, ie last token
                new_examples['stream'].append(new_stream)
                new_examples['patient_id'].append(pt_id)
 
    return new_examples



def create_risk_prediction_timelines(examples, prefixes, token_type, n_risk=1, min_past_length=1, max_timeline_len=None):
    r'''
        max_timeline_len is in number of concepts, not seq_len
        min_past_length - also number of concepts
        n_risk: If there are multiple prefixes one example can be used to generated multiple risk prediction timelines, this determines how many.
    '''
    new_examples = {'patient_id': [], 'stream': []}

    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        pt_id = examples['patient_id'][i]

        time_sep = np.median([x['time'] for x in stream])
        max_conf_cuis = len(set([x['cui'] for x in stream]))
        current_time = None
        _prefixes = list(prefixes) # Copy
        for _ in range(n_risk):
            new_stream = []
            risk_concepts = []
            old_conf_cuis = set()
            one_new = None
            over_time_limit = False
            start_risk = False
            for ent in stream:
                # Do this until we are past the middle, or less than min, or more than max
                if (ent['time'] < time_sep or len(new_stream) <= min_past_length) and (max_timeline_len is None or len(new_stream) < max_timeline_len):
                    new_stream.append(ent)
                    if ent['token_type'] == token_type: # To avoid hypo and negated
                        old_conf_cuis.add(ent['cui'])
                    current_time = ent['time'] # Time of the last token, before we start doing risk prediction
                elif not start_risk:
                    # Go until we find the first new token_type
                    if ent['token_type'] == token_type and ent['cui'] not in old_conf_cuis:
                        start_risk = True
                    else:
                        new_stream.append(ent)
                        current_time = ent['time'] # Time of the last token, before we start doing risk prediction

                if start_risk and ent['token_type'] == token_type:
                    if one_new is None:
                        one_new = ent
                        
                        # Random select an element from the _prefixes array and remove it from _prefixes
                        el = random.choice(_prefixes)
                        _prefixes.remove(el)
                        timediff, prefix = el

                    if ent['time'] <= current_time + timediff:
                        if ent['cui'] not in old_conf_cuis and ent['cui'] not in risk_concepts:
                            risk_concepts.append(ent['cui'])
                    else:
                        # We are over the timelimit, this is a requirement, otherwise timeline is shorter than the timelimit
                        over_time_limit = True
                        break

            ent = stream[-1] 
            if one_new is not None and risk_concepts and len(risk_concepts) > 1 and len(new_stream) >= min_past_length and over_time_limit:
                one_new['token'] = one_new['cui'] = " ".join(risk_concepts)
                one_new['cntx_left'] = prefix.split(" ")
                one_new['cntx_left_inds'] = list(range(len(one_new['cntx_left'])))
                one_new['doc_id'] = ent['doc_id'] # Same as the last token
                one_new['cntx_right'] = []
                one_new['cntx_right_inds'] = []
                one_new['presence'] = None
                one_new['time'] = current_time + 10 # Just move the current time a bit
                one_new['token_type'] = f'risk-{timediff}-{token_type}'
            
                new_stream.append(one_new)
                new_stream.append(ent) # Append the </s> token, ie last token
                #examples['stream'][i] = new_stream
                new_examples['stream'].append(new_stream)
                new_examples['patient_id'].append(pt_id)
 
    return new_examples

def fix_types_for_presence(examples, mapping):
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]

        for ent in stream:
            if ent['presence'] is not None and ent['presence']:
                prefix = mapping[ent['presence']]
                if prefix:
                    ent['token_type'] = f'{prefix}_{ent["token_type"]}'

    return examples


def add_age(examples, pt2info, first_age_format=' {} year old', other_age_format=' at the age of {}', age_normalizer=365.25 * 24 * 60 * 60):
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        pt = examples['patient_id'][i]
        last_age_added = -1
        new_stream = []
        for ent in stream:
            if pt in pt2info:
                age = None
                if pt2info[pt]['DOB'] is not None:
                    age = int((ent['time'] - pt2info[pt]['DOB'].timestamp()) / age_normalizer)

                # Age comes a step before the token that caused the change
                if age is not None and age >= 0 and last_age_added != age:
                    if last_age_added == -1:
                        # Means this is the first age
                        new_stream.append(make_example(ent_example=ent, token=first_age_format.format(age), token_type='age', cnt=10**6, time=ent['time']-1))
                    else:
                        new_stream.append(make_example(ent_example=ent, token=other_age_format.format(age), token_type='age', cnt=10**6, time=ent['time']-1))
                    last_age_added = age
            new_stream.append(ent)
        examples['stream'][i] = new_stream
        new_stream = []

    return examples

def add_ttd(examples, pt2dod_timestamp, ttd_prefix='<TTD>', ttd_suffix=None, ttd_normalizer=365.25 * 24 * 60 * 60,
            max_ttd=10, ttd_prob=1, max_nttd=10, duplicate_streams=False):
    all_patient_id = []
    all_stream = []
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        last_ttd_added = -1
        new_stream = []
        new_streams = [new_stream]
        n_added_ttds = 0
        for ent in stream:
            if examples['patient_id'][i] in pt2dod_timestamp:
                if n_added_ttds < max_nttd:
                    if random.random() <=  ttd_prob:
                        ttd = int((pt2dod_timestamp[examples['patient_id'][i]] - ent['time']) / ttd_normalizer) + 1
                        if ttd <= max_ttd:
                            if last_ttd_added != ttd:
                                if duplicate_streams:
                                    # At this point we duplicate the first stream fron new_streams (it is the one without TTD always)
                                    new_stream = copy.deepcopy(new_streams[0])
                                    new_streams.append(new_stream)

                                if ttd_prefix is not None:
                                    new_stream.append(make_example(ent_example=ent, token=ttd_prefix, token_type='ttd_prefix', cnt=10**6, time=ent['time']))
                                new_stream.append(make_example(ent_example=ent, token=str(ttd), token_type='ttd', cnt=10**6, time=ent['time']))

                                last_ttd_added = ttd
                                if ttd_suffix is not None:
                                    new_stream.append(make_example(ent_example=ent, token=ttd_suffix, token_type='ttd_suffix', cnt=10**6, time=ent['time']))
                                n_added_ttds += 1

            # append the entity to each stream
            for new_stream in new_streams: new_stream.append(ent)

        if duplicate_streams and len(new_streams) > 1:
            # Remove the first example as it is the base one without time info
            del new_streams[0]

        for new_stream in new_streams:
            all_stream.append(new_stream)
            all_patient_id.append(examples['patient_id'][i])

    examples['patient_id'] = all_patient_id
    examples['stream'] = all_stream

    return examples

def split_stream(examples, max_seq_len=-1):
    if max_seq_len > 0:
        new_streams = []
        new_patient_ids = []
        for ind, stream in enumerate(examples['stream']):
            nparts = math.ceil(len(stream) / max_seq_len)
            for i in range(nparts):
                new_streams.append(stream[i*max_seq_len:(i+1)*max_seq_len])
                new_patient_ids.append(examples['patient_id'][ind])

        examples['stream'] = new_streams
        examples['patient_id'] = new_patient_ids

    return examples


def cleanup_stream(examples,  add_context=False, separator=None, ends=['.']):
    r''' Leave only Tokens and remove the rest from `stream`

    Args:
        examples
        keep_time:
            If set another value will be added to examples that contains the `time` for each
            entity in stream.
        keep_type:
            Same as above
    '''
    if 'token' in examples['stream'][0][0]:
        if add_context:
            new_examples = {}
            new_examples['stream'] = []
            new_examples['time'] = []
            new_examples['token_type'] = []
            for stream in examples['stream']:
                # End cannot be bigger than next ent ind
                # Start has to be bigger than existing inds
                prev_doc_id = -1
                new_tokens = []
                new_time = []
                new_token_type = []
                last_ind = -1
                last_token_text = ''
                last_token_type = 'text'
                for ent_ind, ent in enumerate(stream):
                    start_of_new_doc = False
                    # If the previous doc_id is not equal to current doc_id,
                    #means this is a new document and last_ind can be rest
                    if prev_doc_id != ent['doc_id']:
                        last_ind = -1
                        prev_doc_id = ent['doc_id']
                        start_of_new_doc = True

                    # If last entity in stream, or last entity in document (next entity is new doc)
                    if ent_ind == len(stream) - 1:
                        next_end_ind = 10**9
                    elif stream[ent_ind + 1]['doc_id'] != ent['doc_id']:
                        next_end_ind = 10**9
                    else:
                        # This is not the last entity from this doc, so we take the
                        #id of the next entity token
                        _id = stream[ent_ind + 1]['ent_tkn_id']
                        next_end_ind = _id if _id is not None else 10**9
                    
                    def _add_token(tkn, arr):
                        # This ruins some of the effort on keeping multiple spaces as they might be important, but it is just too hard
                        tkn = tkn.strip(' ')
                        """
                        if arr:
                            if len(arr[-1]) == 0 or arr[-1][-1] not in {' ', '\n', '\r', '\t'}:
                                if tkn and tkn[0] != ' ':
                                    tkn = ' ' + tkn
                        """
                        arr.append(tkn)

                    if separator is not None and last_ind > 0 and (len(ent['cntx_left_inds']) == 0 or ent['cntx_left_inds'][0] > (last_ind + 5)) and ent['token_type'] != "eos_token":
                        # We add the separator as the context of this concept does not
                        #continue the context of the previous concept

                        # Only append if the previous does not end with any of sentence ends (e.g. .)
                        if last_token_text and not any([str(last_token_text).strip().endswith(end) for end in ends]):
                            _add_token(separator, new_tokens)
                            new_time.append(ent['time'])
                            new_token_type.append('snt-sep')
                    elif separator is not None and start_of_new_doc and last_token_text:
                        # In case the document changed, but it does not end with '.'
                        if not any([str(last_token_text).strip().endswith(end) for end in ends]):
                            _add_token(separator, new_tokens)
                            new_time.append(ent['time'])
                            new_token_type.append('snt-sep')

                    # Add left context
                    if ent['cntx_left'] is not None and ent['cntx_left']:
                        for tkn_ind, tkn in zip(ent['cntx_left_inds'], ent['cntx_left']):
                            if tkn_ind > last_ind:
                                _add_token(tkn, new_tokens)
                                new_time.append(ent['time'])
                                new_token_type.append('text')
                    # Add entity itself
                    _add_token(ent['token'], new_tokens)
                    new_time.append(ent['time'])
                    new_token_type.append(ent['token_type'])
                    last_ind = ent['ent_tkn_id'] if ent['ent_tkn_id'] is not None else -1
                    last_token_text = ent['token']
                    last_token_type = 'cui'
                    # Add right context
                    if ent['cntx_right'] is not None and ent['cntx_right']:
                        for tkn_ind, tkn in zip(ent['cntx_right_inds'], ent['cntx_right']):
                            if tkn_ind < next_end_ind:
                                _add_token(tkn, new_tokens)
                                new_time.append(ent['time'])
                                new_token_type.append('text')
                                last_ind = tkn_ind
                                last_token_text = tkn
                                last_token_type = 'text'
                    
                new_examples['stream'].append(new_tokens)
                new_examples['time'].append(new_time)
                new_examples['token_type'].append(new_token_type)
            examples['stream'] = new_examples['stream']
            examples['time'] = new_examples['time']
            examples['token_type'] = new_examples['token_type']
        else:
            # No context
            new_examples = {}
            new_examples['time'] = []
            new_examples['token_type'] = []
            new_examples['stream'] = []
            for i in range(len(examples['stream'])):
                times = []
                token_types = []
                streams = []
                for j in range(len(examples['stream'][i])):
                    # When there is no context we do not use negative/hypothetical stuff
                    if not (examples['stream'][i][j]['token_type'].startswith('Hypothetical_T') or examples['stream'][i][j]['token_type'].startswith('No_T')):
                        times.append(examples['stream'][i][j]['time'])
                        token_types.append(examples['stream'][i][j]['token_type'])
                        streams.append(examples['stream'][i][j]['token'].strip())
                new_examples['time'].append(times)
                new_examples['token_type'].append(token_types)
                new_examples['stream'].append(streams)

            #examples['time'] = [[ent['time'] for ent in stream] for stream in examples['stream']]
            #examples['token_type'] = [[ent['token_type'] for ent in stream] for stream in examples['stream']]
            #examples['stream'] = [[ent['token'].strip(' ') for ent in stream] for stream in examples['stream']]
            examples = new_examples
    return examples

def add_to_stream(examples, pt2info=None, one_token=None, last=False, prefix=None, key=None, unk_tkn='Unknown', token_type='unk', lowercase=False, add_space=True):
    r''' Add information to the patient stream based on patient_id.

    Args:
        examples
        pt2tkn
        last
        unk_tkn:
            What token will be added if the patient_id is not in pt2tkn
    '''

    for i in range(len(examples['stream'])):
        ent = examples['stream'][i][0]
        pt = examples['patient_id'][i]
        if (pt2info and pt in pt2info) or one_token:
            if one_token:
                token = one_token
            else:
                token = pt2info[pt][key]
                token = token if token is not None else unk_tkn
                token = token.lower() if lowercase else token

            t_ind = -1 if last else 0 # If -1 means it is the last token, otherwise the first
            temporal_difference = -1 if last else 1
            prepend = ' ' if add_space else ''
            to_append = [make_example(ent_example=ent, token=token, cnt=10**6, time=examples['stream'][i][t_ind]['time'] - temporal_difference, token_type=token_type,
                                      prefix=prepend)]
            if prefix is not None:
                prefix_token = make_example(ent_example=ent, token=prefix, cnt=10**6,
                                            time=examples['stream'][i][t_ind]['time'], token_type="prefix_" + token_type)
                to_append = [prefix_token] + to_append

            if last:
                # Append as last token
                examples['stream'][i] = examples['stream'][i] + to_append
            else:
                examples['stream'][i] = to_append + examples['stream'][i]

    return examples


def remove_tokens_not_in_tokenizer(examples, tokens_to_keep):
    tokens_to_keep = set(tokens_to_keep)
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        new_stream = []

        for ent in stream:
            tkn = ent['token']

            if tkn in tokens_to_keep:
                new_stream.append(ent)

        examples['stream'][i] = new_stream

    return examples


def remove_parents_from_stream(examples, ch2parents, separator=None, separators=None):
    for i in range(len(examples['stream'])):
        stream = examples['stream'][i]
        parents = set()
        new_stream = []

        for ent in stream:
            tkn = ent['cui']

            if (separator is not None and tkn == separator) or (separators is not None and tkn in separators):
                # This means we are removing parents only inside of one bucket
                parents = set()

            if tkn and tkn in ch2parents and ent['presence'] == 'True':
                # Add only if not in parents
                if tkn not in parents:
                    new_stream.append(ent)
                # Update parents
                parents.update(ch2parents[tkn])
            else:
                new_stream.append(ent)

        examples['stream'][i] = new_stream

    return examples

def get_embeddings_for_tokens(dataset=None, cdb=None, context_type='medium', normalize=True, extra_tokens=['<PAD>'], types=None, concepts=None):
    r''' Given a stream of tokens get the embeddings from MedCAT and make the required maps.

    Args:
        dataset
        cdb
        context_type
        normalize:
            If True the embedding vectors will be normalized
        tkn2type:
            Dictionary mapping from token to type
        types:
            All posible token types (e.g. [T-11, T-12, ...]
        concepts:
            If provided these concepts will also be appened to the tokens and supported by the tokenizer
    Returns:
        embeddings
        tkn2id
        id2tkn
        id2type
        id2type_detailed
    '''
    embeddings = []
    tkn2id = {}
    id2tkn = {}

    def add_tkn(tkn):
        if tkn in cdb.cui2context_vectors and context_type in cdb.cui2context_vectors[tkn]:
            vec = cdb.cui2context_vectors[tkn][context_type]
        else:
            # Token vector is randomly assigned
            vec = np.random.rand(300)

        id2tkn[len(embeddings)] = tkn
        tkn2id[tkn] = len(embeddings)

        vec = unitvec(vec) if normalize else vec
        embeddings.append(vec)

    datasets = get_all_splits(dataset)
    for _dataset in datasets:
        for stream in _dataset['stream']:
            for tkn in stream:
                tkn = str(tkn)
                if tkn not in tkn2id:
                    add_tkn(tkn)
    # Add concepts if they are provided, this is used to build a general
    #tokenizer with all concepts
    if concepts is not None:
        for concept in concepts:
            tkn = str(concept)
            if tkn not in tkn2id:
                add_tkn(tkn)

    # Add named tokens
    for tkn in extra_tokens:
        if tkn not in tkn2id:
            id2tkn[len(embeddings)] = tkn
            tkn2id[tkn] = len(embeddings)
            if tkn != '<PAD>':
                embeddings.append(np.random.rand(len(embeddings[0])))
            else:
                embeddings.append(np.zeros(len(embeddings[0])))

    # Add type tokens
    for tkn in types:
        if tkn not in tkn2id:
            id2tkn[len(embeddings)] = tkn
            tkn2id[tkn] = len(embeddings)
            embeddings.append(np.random.rand(len(embeddings[0])))

    return embeddings, tkn2id, id2tkn


def stream_to_separate_examples(examples):
    r''' Convert a stream to separate examples that can be used to train
    a next concept predictor unable to handle sequences (e.g. random forset). Use with HF datasets map function.

    '''
    out = {}
    out['input_ids'] = [input_ids[0:i+1] for input_ids in examples['input_ids'] for i in range(len(input_ids) - 1)]
    out['labels'] = [input_ids[i+1] for input_ids in examples['input_ids'] for i in range(len(input_ids) - 1)]
    out['labels_all'] = [input_ids[i+1:] for input_ids in examples['input_ids'] for i in range(len(input_ids) - 1)]
    out['patient_id'] = [patient_id for ind, patient_id in enumerate(examples['patient_id']) for _ in range(len(examples['input_ids'][ind]) - 1)]

    return out
