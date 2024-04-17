from box import Box
import jsonpickle
import os
import yaml

class BaseConfig(object):
    def __init__(self, to_box=False):
        pass

    def _to_box(self):
        # Convert all dicts to boxes
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                self.__setattr__(key, Box(val))

    def _from_box(self):
        # Convert all dicts to boxes
        for key, val in self.__dict__.items():
            if isinstance(val, Box):
                self.__setattr__(key, val.to_dict())

    def save(self, save_path=None):
        r''' Save the config into a .json file
        Args:
            save_path (`str`):
                Where to save the created json file, if nothing we use the default from paths.
        '''
        if save_path is None:
            save_path = self.path.self

        # We want to save the dict here, not the whole class
        self._from_box()
        json_string = jsonpickle.encode({k:v for k,v in self.__dict__.items() if k != 'path'})

        with open(save_path, 'w') as f:
            f.write(json_string)
        self._to_box()

    @classmethod
    def load(cls, save_path):
        config = cls(to_box=False)
        # Read the jsonpickle string
        with open(save_path) as f:
            config_dict = jsonpickle.decode(f.read())
        config.merge_config(config_dict)
        config._to_box()
        return config

    def merge_config(self, config_dict):
        r''' Merge a config_dict with the existing config object.
        Args:
            config_dict (`dict`):
                A dictionary which key/values should be added to this class.
        '''
        for key in config_dict.keys():
            if key in self.__dict__ and isinstance(self.__dict__[key], dict):
                self.__dict__[key].update(config_dict[key])
            else:
                self.__dict__[key] = config_dict[key]


class Config(BaseConfig):
    r''' There are probably nicer ways to do this, but this one works for this use-case.
    '''
    def __init__(self, yaml_path, extra_yaml_paths=None):
        self.yaml_path = yaml_path
        if isinstance(extra_yaml_paths, str):
            extra_yaml_paths = [extra_yaml_paths]
        self.extra_yaml_paths = extra_yaml_paths
        self.load_yaml(yaml_path)
        if extra_yaml_paths:
            for extra_yaml_path in extra_yaml_paths:
                self.load_yaml(extra_yaml_path)

    def reload_yaml(self):
        self.load_yaml(self.yaml_path)
    
    def merge_dicts(self, A, B):
        for key in B:
            if key in A:
                if isinstance(A[key], dict) and isinstance(B[key], dict):
                    self.merge_dicts(A[key], B[key])
                else:
                    A[key] = B[key]
            else:
                A[key] = B[key]

    def load_yaml(self, yaml_path):
        self._config = yaml.safe_load(open(yaml_path, 'r'))
        self.merge_dicts(self.__dict__, self._config)

        if hasattr(self, 'to_box') and self.to_box:
            self._to_box()

            def create_dirs(paths):
                for path in paths:
                    if isinstance(path, str):
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                    elif isinstance(path, dict):
                        create_dirs(path.values())
            create_dirs(self.path.values())
    
    @property
    def path(self):
        r''' This will rerun the f-strings each time, who cares. One could do it only when something was updated, but
        again, who cares.
        '''
        use_snt_limits = False
        if hasattr(self.train, 'sentence_limits') and self.train.sentence_limits is not None:
            use_snt_limits=True

        train_run_name = f'{self.id}-cntx_size_{self.train.cntx_size}-{self.train.dataset_name}-train_days_{self.train.days}-train_types_{"_".join(self.train.types)}'
        tokenizer_path = f'{self.base_path}/models/foresight/{self.dataset}/{self.model.base_name.split("/")[-1]}-tokenizer/'
        ds_extra = f'use_snt_limits_{use_snt_limits}-use_context_{self.train.use_context}'
        train_extra = f'max_timeline_len_{self.train.max_timeline_len}'

        ds_path = f'{self.base_path}/data/{self.dataset}'
        train_path = f'{ds_path}/{self.train.base_name}'
        
        self._path = {
                'self': f'{ds_path}/{self.train.base_name}/config_for_{train_run_name}.json',
                'model': f'{self.base_path}/models/{self.dataset}/{self.model.base_name.split("/")[-1]}',
                'trained_model': f'{self.base_path}/models/{self.dataset}/{train_run_name}_{ds_extra}_{train_extra}_{self.model.base_name.split("/")[-1]}',
                'trained_model_risk': f'{self.base_path}/models/{self.dataset}/{train_run_name}_risk_{ds_extra}_{train_extra}_{self.model.base_name.split("/")[-1]}',
                'tokenizer': {
                   'self': tokenizer_path,
                    'tkn2type': os.path.join(tokenizer_path, 'tkn2type.pickle'),
                    'tkn_id2type': os.path.join(tokenizer_path, 'tkn_id2type.pickle'),
                    'id2tkn': os.path.join(tokenizer_path, 'id2tkn.pickle'),
                    'token_type2tokens': os.path.join(tokenizer_path, 'token_type2tokens.pickle'),
                },
                'dataset': {
                    'pt2info': f'{ds_path}/pt2info.pickle',
                    'doc2text': f'{ds_path}/doc2text.pickle',
                    'doc2info': f'{ds_path}/doc2info.pickle',
                    'train_df': f'{ds_path}/prepared_llm/train_noteevents_with_codes.csv',
                    'text_with_codes_prepared': f'{ds_path}/llm_all_text_dataset.hf',

                    'text_with_codes': f'{train_path}/text_with_codes.csv',
                    'annotated_documents': f'{train_path}/annotated_documents/',
                    'cuis_in_text': f'{train_path}/prepared_llm/cuis_in_text.pickle',
                    'test_df': f'{train_path}/prepared_llm/test_noteevents_with_codes.csv',
                    'test_sets_folder': f'{train_path}/test_sets/',

                    'cui_by_pt': f'{train_path}/{self.id}_{self.train.dataset_name}_cui_by_pt.pickle',
                    'pt2cui2cnt': f'{train_path}/{self.id}_{self.train.dataset_name}_pt2cui2cnt.pickle',

                    'self': f'{train_path}/{self.id}_{self.train.dataset_name}-cntx_size_{self.train.cntx_size}-use_snt_limits_{use_snt_limits}.pickle',
                    'splits_data':  f'{train_path}/{self.id}_{self.train.dataset_name}-cntx_size_{self.train.cntx_size}-use_snt_limits_{use_snt_limits}-splits_data/',
                    'prepared_dataset_split': f'{train_path}/{train_run_name}-{ds_extra}-prepared_dataset/',
                    'prepared_risk_dataset': f'{train_path}/{train_run_name}-{ds_extra}-prepared_risk_dataset/',
                    'just_before_encoding_dataset_split': f'{train_path}/{train_run_name}-{ds_extra}-just_before_encoding/',
                    'just_before_training_dataset_split': f'{train_path}/{train_run_name}-{ds_extra}-{train_extra}-just_before_training/',
                    
                    'metrics_folder': f'{train_path}/{train_run_name}-{ds_extra}-{train_extra}-{self.model.base_name.split("/")[-1]}-metrics/',
                    'hf_output_folder': f'{train_path}/{train_run_name}-{ds_extra}-{train_extra}-{self.model.base_name.split("/")[-1]}-hf-output/',
                    },
                }
        # Some paths are static
        def recursive_update(dest_dict, src_dict):
            for key, value in src_dict.items():
                if isinstance(value, dict) and key in dest_dict and isinstance(dest_dict[key], dict):
                    recursive_update(dest_dict[key], value)
                else:
                    dest_dict[key] = value
        recursive_update(self._path, self.static_paths)

        return Box(self._path)
