from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch

InputDataClass = NewType("InputDataClass", Any)

class CollataAndPad(object):
    r''' Arrange the data into the right format + add padding or trim where necessary. Trips from the RIGHT

    Args:
        max_seq_len (`int`, `optional`, defaults to -1):
            Upper bound for sequence length. If it is -1 means that it will be
            calculated for each bach and set to the max length without upper limits.
        pad_id (`int`, `optional`, defaults to 0):
            What ID will be used to pad the inputs to max_seq_len
    '''
    def __init__(self, max_seq_len=-1, pad_id=0):
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id


    def __call__(self, features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        batch = {}
        if self.max_seq_len == -1:
            max_seq_len = max([len(f['input_ids']) for f in features])
        else:
            max_seq_len = min(self.max_seq_len, max([len(f['input_ids']) for f in features]))


        batch['labels'] = torch.tensor([f['labels'][:max_seq_len] + [-100] * max(0, max_seq_len-len(f['labels'])) 
                                        for f in features], dtype=torch.long)
        batch['input_ids'] = torch.tensor([f['input_ids'][:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(f['input_ids']))
                                          for f in features], dtype=torch.long)
        if 'position_ids' in batch:
            # Padding for position ids is max_seq_len - 1
            batch['position_ids'] = torch.tensor([f['position_ids'][:max_seq_len] + [max_seq_len - 1] * max(0, max_seq_len - len(f['position_ids']))
                                          for f in features], dtype=torch.long)

        if 'time' in batch: 
            # Padding for position ids is max_seq_len - 1
            batch['time'] = torch.tensor([f['time'][:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(f['time']))
                                          for f in features], dtype=torch.long)
        if 'token_type' in batch: 
            # Padding for position ids is max_seq_len - 1
            batch['token_type'] = torch.tensor([f['token_type'][:max_seq_len] + [self.pad_id] * max(0, max_seq_len - len(f['token_type']))
                                          for f in features], dtype=torch.long)

        #batch['attention_mask'] = batch['input_ids'] != self.pad_id

        return batch
