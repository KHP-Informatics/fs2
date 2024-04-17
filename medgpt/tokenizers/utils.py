import torch
import math

def encode_stream(examples, tokenizer):
    # This got more complicated then expected, but fuck it, it is one of the last steps
    #at this point the dataset is usually not that big so should be OK.
    input_ids = []
    attention_mask = []
    time = []
    token_type = []
    for ind, stream in enumerate(examples['stream']):
        _ids = []
        _mask = []
        _time = []
        _token_type = []
        # Here we say that text is not split into words, even though it is, so that I can get
        #the parts/tokens of each words and assign time and token type.
        encoded = tokenizer(stream, add_special_tokens=False, is_split_into_words=False)
        for i in range(len(encoded['input_ids'])):
            _ids.extend(encoded['input_ids'][i])
            _mask.extend(encoded['attention_mask'][i])
            _time.extend([examples['time'][ind][i]] * len(encoded['input_ids'][i]))
            _token_type.extend([examples['token_type'][ind][i]] * len(encoded['input_ids'][i]))
        input_ids.append(_ids)
        attention_mask.append(_mask)
        time.append(_time)
        token_type.append(_token_type)
    examples['input_ids'] = input_ids
    examples['attention_mask'] = attention_mask
    examples['time'] = time
    examples['token_type'] = token_type

    return examples

def pack_text(examples, max_len):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Discard the examples that do not make a full chunk
    total_length = (total_length // max_len) * max_len
    # Split by chunks of max_len.
    result = {
        k: [torch.tensor(t[i:i + max_len]) for i in range(0, total_length, max_len)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def trim_ds(examples, max_len):
    return {k:[one_v[0:max_len] for one_v in v] for k,v in examples.items()}

def partial_pack_for_risk(examples, max_len):
    result = {k:[] for k in examples.keys()}
    _key = list(examples.keys())[0] # Take whichever key
    new_example = {k:[] for k in examples.keys()}

    for ind in range(len(examples[_key])):
        # Trim long sequences to max_len, from the right side (ie remove left side tokens)
        example = {k:v[ind][-max_len:] for k,v in examples.items()}
        if len(new_example[_key]) + len(example[_key]) > max_len:
            result = {k:result[k] + [v] for k,v in new_example.items()}
            new_example = example 
        else:
            new_example = {k:new_example[k] + v for k,v in example.items()}
    #  Add the last example if there is something to add  
    if len(new_example[_key]) > 0:   
        result = {k:result[k] + [v] for k,v in new_example.items()}
    
    return result


def pack_examples(examples, block_size, packing_type='partial', split_long_examples=True):
    r''' Used with a prepared HF dataset, will pack/group examples. Use with care, can mess up many things
    if the input is not formated properly (requires the <|eod|> token).
    
    packing_type: partial/full/no 
    split_long_examples: If true long examples will be split into multiple, makes sense only for partial packing, full packing will always have this as True
    '''
    # Concatenate all texts.
    if packing_type == 'partial':
        result = {k:[] for k in examples.keys()}
        _key = list(examples.keys())[0] # Take whichever key
        new_example = {k:[] for k in examples.keys()}

        for ind in range(len(examples[_key])):
            # Trim long sequences to block_size, this is required for partial packing
            example = {k:v[ind][0:block_size] for k,v in examples.items()}
            if len(new_example[_key]) + len(example[_key]) > block_size:
                result = {k:result[k] + [v] for k,v in new_example.items()}
                new_example = example 
            else:
                new_example = {k:new_example[k] + v for k,v in example.items()}
        #  Add the last example if there is something to add  
        if len(new_example[_key]) > 0:   
            result = {k:result[k] + [v] for k,v in new_example.items()}
    elif packing_type == 'full':
        # Full packing
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    elif packing_type == 'split':
        # split long examples into max_seq_len
        result = {k:[] for k in examples.keys()}
        _key = list(examples.keys())[0] # Take whichever key

        for ind in range(len(examples[_key])):
            for i in range(len(math.ceil(example[_key]) / block_size)):
                start = i * block_size
                end = (i+1) * block_size
                result = {k:result[k] + [v[start:end]] for k,v in example.items()}
    else:
        # Do nothing
        result = examples

    result["labels"] = result["input_ids"].copy()
    return result

def create_labels(examples, config, cui_ids=None, type_names=None, copy_input_ids=False, extra_label_ids=None, max_seq_len=None, r_trim=False):
    r''' This can be used to create a dataset where only the CUIs will be trained on, nothing else
    '''
    examples['labels'] = []
    if max_seq_len:
        if not r_trim:
            examples = {k:[one_v[0:max_seq_len] for one_v in v] for k,v in examples.items()}
        else:
            examples = {k:[one_v[-max_seq_len:] for one_v in v] for k,v in examples.items()}

    if not copy_input_ids:
        for i in range(len(examples['input_ids'])):
            labels = []
            if cui_ids:
                for tkn_id in examples['input_ids'][i]:
                    if tkn_id in cui_ids or (extra_label_ids and tkn_id in extra_label_ids):
                        labels.append(tkn_id)
                    else:
                        labels.append(config.train.ignore_index)
                examples['labels'].append(labels)
            elif type_names:
                for tkn_type, tkn_id in zip(examples['token_type'][i], examples['input_ids'][i]):
                    if tkn_type in type_names or (extra_label_ids and tkn_id in extra_label_ids):
                        labels.append(tkn_id)
                    else:
                        labels.append(config.train.ignore_index)
                examples['labels'].append(labels)
    else:
        examples["labels"] = examples["input_ids"].copy()
    return examples
