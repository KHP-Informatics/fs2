r'''

My apologies, this is a mess, but it does work.

'''
import os
import sys
sys.path.insert(0, "/data/zeljko/projects/medgpt/")
sys.path.insert(0, "/data/zeljko/projects/MedCAT/")

os.environ['HF_DATASETS_CACHE'] = "/data/zeljko/.cache/huggingface"
os.environ['TRANSFORMERS_CACHE'] = "/data/zeljko/.cache/huggingface"

import argparse
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from medgpt.tokenizers.utils import pack_text, create_labels, pack_examples
import pickle
from medcat.cat import CAT
import pandas as pd
import datasets
import random
import math
from medgpt.config import Config
from transformers import BitsAndBytesConfig
import json
from medgpt.metrics.next_concept_prediction import ComputePrecisionHF
from datasets import Dataset
from medgpt.datasets.data_collator import CollataAndPad
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config')
parser.add_argument('-xc', '--xconfig')
parser.add_argument('-nproc', '--nproc', default=16)
args = parser.parse_args()

config = Config(yaml_path=args.config, extra_yaml_paths=args.xconfig)

cat = CAT.load_model_pack(config.path.cat, meta_cat_config_dict={'general': {'device': config.cat.meta.device}})
cdb = cat.cdb

tokenizer = AutoTokenizer.from_pretrained(config.path.tokenizer.self)
bnb_config = BitsAndBytesConfig(
   load_in_8bit=True,
)
model = AutoModelForCausalLM.from_pretrained(config.path.model, use_flash_attention_2=True)#, torch_dtype=torch.bfloat16)#quantization_config=bnb_config)#, device_map='auto')

try:
    dataset = datasets.load_from_disk(config.path.dataset.just_before_training_dataset_split)
    print("Loaded an existing dataset")
except Exception as e:
    print(e)
    print("Building a new DS")
    dataset = datasets.load_from_disk(config.path.dataset.prepared_dataset_split)
    dataset['train'] = dataset['train'].remove_columns(['patient_id', 'token_type', 'time'])
    dataset['test'] = dataset['test'].remove_columns(['patient_id', 'token_type', 'time'])
    
    dataset = dataset.map(
        lambda examples: pack_text(examples, max_len=config.train.max_timeline_len),
        batched=True,
        batch_size=1000,
        num_proc=args.nproc,
    )

    # Create labels for supervised training
    cuis = pickle.load(open(config.path.dataset.cuis_in_text, 'rb'))
    cui_ids = set(tokenizer.convert_tokens_to_ids([c for c in cuis]))
    dataset = dataset.map(
        lambda examples: create_labels(examples, config, cui_ids),
        batched=True,
        batch_size=1000,
        num_proc=args.nproc,
    )
    dataset.save_to_disk(config.path.dataset.just_before_training_dataset_split)

print("Will start training now")
targs = config.train.hf_training_arguments.to_dict()
# Set the dynamic dir for output
targs['output_dir'] = config.path.dataset.hf_output_folder
training_args = TrainingArguments(**targs)

test_dataset = datasets.load_from_disk(config.path.dataset.prepared_dataset_split)['test']
test_dataset = test_dataset.remove_columns(['patient_id'])

mini_train = datasets.Dataset.from_dict(dataset['train'][random.sample([i for i in range(len(dataset['train']))], k=1000)])
mini_eval = datasets.Dataset.from_dict(test_dataset[random.sample([i for i in range(len(test_dataset))], k=config.train.mini_eval_size)])
# Add labels, if not added loss makes no sense but metrics are still fine
cuis = pickle.load(open(config.path.dataset.cuis_in_text, 'rb'))
cui_ids = set(tokenizer.convert_tokens_to_ids([c for c in cuis]))
mini_eval = mini_eval.map(
    lambda examples: create_labels(examples, config, cui_ids, max_seq_len=config.train.max_timeline_len),
        batched=True,
        batch_size=1000,
        num_proc=args.nproc,
)

token_type2tokens = pickle.load(open(config.path.tokenizer.token_type2tokens, 'rb'))
id2tkn = {v:k for k,v in tokenizer.vocab.items()}
compute_metrics = ComputePrecisionHF(id2tkn, 
                                     prediction_scope='time_range', 
                                     topk=1, # 1, 5, 10
                                     start=0, # 0, 10, 20, 50, 100
                                     return_all_metrics=False, 
                                     batch_size=10, 
                                     select_token_types=set(token_type2tokens.keys()),
                                     type_data=mini_eval['token_type'],
                                     token_type2tokens=token_type2tokens,
                                     time_data=mini_eval['time'],
                                     time_range=30*24*60*60, #30, 365, 1000000
                                     ignore_label_status=False,
                                     min_time_left=24*60*60,)

#dc = CollataAndPad(max_seq_len=config.train.max_timeline_len, pad_id=tokenizer.pad_token_id)
print(config.train.max_timeline_len)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=mini_eval,
    compute_metrics=compute_metrics,
#    data_collator=dc,
)
trainer.train()
trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
trainer.save_model(config.path.trained_model)