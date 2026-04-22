import os
import gzip
import json
import random
import logging
import torch
import datasets
import transformers
from transformers import GenerationConfig
from tqdm import tqdm

from dataclasses import dataclass
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset, PROMPT_DICT
from federatedscope.core.data.utils import download_url
from federatedscope.llm.model.model_builder import parse_model_type, \
    get_model_cache_dir

logger = logging.getLogger(__name__)


@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class Predictor:
    """Generate the output from the original LLM model"""
    def __init__(self, config, tokenizer, generate_kwargs=None):
        self.device = f'cuda:{config.device}'

        # self.model = get_llm(config).to(self.device)
        self.add_special_tokens = True
        self.tokenizer = tokenizer

        if generate_kwargs is not None:
            self.generate_kwargs = generate_kwargs
        else:
            self.generate_kwargs = {
                'max_new_tokens': config.llm.chat.max_len,
                'num_beams': 4,
                'no_repeat_ngram_size': 2,
                'early_stopping': True,
                'temperature': 0.0
            }

    def __call__(self, input_text, model):
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0).to(self.device)
        response = model.generate(input_ids=input_ids, **self.generate_kwargs)
        response_tokens = \
            self.tokenizer.decode(response[0][input_ids.shape[1]:],
                                  skip_special_tokens=True)
        if response_tokens == "":
            print('INPUT:', input_text)
            print(len(input_text))
            print('===============================\n\n')
        return response_tokens


def get_tokenizer(model_name, cache_dir, tok_len=128):
    from transformers import AutoTokenizer

    model_name = os.path.expanduser(model_name)
    tokenizer_kwargs = dict(model_max_length=tok_len,
                            padding_side="right",
                            use_fast=False)
    if os.path.exists(model_name):
        tokenizer_kwargs['local_files_only'] = True
    elif cache_dir:
        tokenizer_kwargs['cache_dir'] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value

    num_new_tokens = tokenizer.add_special_tokens(special_tokens)

    return tokenizer, num_new_tokens


class new_dict(dict):
    """
    Create a new_dict to ensure we can access the dictionary with
    one bracket only
    e.g., dict[key1][key2][key3] --> dict[key1.key2.key3]
    """
    def __init__(self, init_dict: dict):
        self.dict = init_dict
        for key in self.dict.keys():
            if type(self.dict[key]) is dict:
                self.dict[key] = new_dict(self.dict[key])
            if type(self.dict[key]) is list:
                self.dict[key] = new_dict({
                    str(idx): value
                    for idx, value in enumerate(self.dict[key])
                })

    def __getitem__(self, __key):
        try:
            if '.' not in __key:
                return self.dict[__key]
            else:
                prefix, suffix = __key.split('.', 1)
                return self.dict[prefix][suffix]
        except:
            return None

    def __setitem__(self, __key, __value):
        if type(__value) is dict:
            self.dict[__key] = new_dict(__value)
        else:
            if '.' not in __key:
                self.dict[__key] = __value
            else:
                prefix, suffix = __key.split('.', 1)
                if prefix not in self:
                    self.dict[prefix] = new_dict({})
                self.dict[prefix][suffix] = __value


def load_json(file_path,
              instruction='instruction',
              input='input',
              output='output',
              category='category',
              **kwargs):
    # Format: [{'instruction': ..., 'input': ..., 'output':...}]
    with open(file_path, 'r', encoding="utf-8") as f:
        list_data_dict = json.load(f)

    # Replace key
    new_list_data_dict = []
    for item in list_data_dict:
        new_item = dict(
            instruction=item[instruction] if instruction in item else None,
            input=item[input] if input in item else None,
            output=item[output] if output in item else None,
            category=item[category] if category in item else None)
        for key, value in kwargs.items():
            new_item[key] = item[value]
        new_list_data_dict.append(new_item)
    return new_list_data_dict


def load_jsonl(file_path,
               is_gzip=False,
               instruction='instruction',
               input='input',
               output='output',
               category='category',
               **kwargs):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, 'r') as f:
        for line in f:
            item = new_dict(json.loads(line))
            new_item = dict(instruction=item[instruction],
                            input=item[input],
                            output=item[output],
                            category=item[category])
            for key, value in kwargs.items():
                new_item[key] = item[value]
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def load_jsonls(file_paths,
                is_gzip=False,
                instruction='instruction',
                input='input',
                output='output',
                category='category',
                **kwargs):
    list_data_dict = []
    for path in file_paths:
        list_data_dict.extend(
            load_jsonl(path, is_gzip, instruction, input, output, category,
                       **kwargs))
    return list_data_dict


def load_llm_dataset(config=None, **kwargs):
    model_name, _ = parse_model_type(config.model.type)
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, get_model_cache_dir(config),
                      config.llm.tok_len)

    if '@' in config.data.type:
        dataset_name, _ = config.data.type.rsplit('@', 1)
    else:
        dataset_name = config.data.type

    def resolve_data_file(file_name):
        if os.path.isabs(file_name) or os.path.exists(file_name):
            return file_name
        return os.path.join(config.data.root, file_name)

    def ensure_data_file(file_name,
                         url=None,
                         rename_from=None,
                         description='dataset'):
        file_path = resolve_data_file(file_name)
        if os.path.exists(file_path):
            return file_path

        if url is not None:
            download_url(url, config.data.root)
            if rename_from is not None:
                src_path = os.path.join(config.data.root, rename_from)
                if os.path.exists(src_path) and not os.path.exists(file_path):
                    os.rename(src_path, file_path)

        if os.path.exists(file_path):
            return file_path

        raise FileNotFoundError(
            f'{description} not found at `{file_path}`. '
            f'Please upload the prepared file to the server first.')

    if dataset_name.endswith('.json'):
        fp = resolve_data_file(dataset_name)
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.endswith('.jsonl'):
        fp = resolve_data_file(dataset_name)
        list_data_dict = load_jsonl(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'alpaca':
        fp = ensure_data_file(
            'alpaca_data.json',
            'https://raw.githubusercontent.com/tatsu-lab'
            '/stanford_alpaca/'
            '761dc5bfbdeeffa89b8bff5d038781a4055f796a/'
            'alpaca_data.json',
            description='alpaca data')
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'alpaca_cleaned':
        fp = ensure_data_file(
            'alpaca_data_cleaned.json',
            'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/'
            'a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/'
            'alpaca_data_cleaned.json',
            description='alpaca cleaned data')
        list_data_dict = load_json(fp)
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'dolly-15k':
        fp = ensure_data_file(
            'databricks-dolly-15k.jsonl',
            'https://raw.githubusercontent.com/databrickslabs'
            '/dolly/d000e3030970379aabbf6d291f50ffdd3b715b64'
            '/data/databricks-dolly-15k.jsonl',
            description='dolly-15k data')
        list_data_dict = load_jsonl(fp,
                                    instruction='instruction',
                                    input='context',
                                    output='response',
                                    category='category')
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'gsm8k':
        fp = ensure_data_file(
            'gsm8k_train.jsonl',
            'https://raw.githubusercontent.com/openai/grade-school-math'
            '/3101c7d5072418e28b9008a6636bde82a006892c/'
            'grade_school_math/data/train.jsonl',
            rename_from='train.jsonl',
            description='gsm8k train data')
        list_data_dict = load_jsonl(fp,
                                    instruction='question',
                                    output='answer')
        for i in range(len(list_data_dict)):
            list_data_dict[i]['output'] = \
                list_data_dict[i]['output'].replace('####', 'The answer is')
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'code_search_net':
        from federatedscope.llm.dataset.code_search_net import \
            CSN_FILE_NUM_DICT

        list_data_dict = []
        logger.info('Loading code search net data file...')
        try:
            for language in tqdm(CSN_FILE_NUM_DICT.keys()):
                sub_list_data_dict = []
                for file_index in range(CSN_FILE_NUM_DICT[language]['train']):
                    fp = \
                        os.path.join(config.data.root, language,
                                     'final', 'jsonl', 'train',
                                     f'{language}_train_{file_index}.jsonl.gz')
                    tmp_list_data_dict = load_jsonl(
                        fp,
                        instruction='docstring',
                        input='language',
                        output='code',
                        category='language',
                        is_gzip=True,
                    )
                    sub_list_data_dict += tmp_list_data_dict
                # Subsample
                raw_size = len(sub_list_data_dict)
                num_subsample = int(raw_size * config.data.subsample)
                list_data_dict += random.sample(sub_list_data_dict,
                                                num_subsample)
                logger.info(f"Subsample "
                            f"{sub_list_data_dict[0]['category']} with "
                            f"rate {config.data.subsample}: "
                            f"the sample size is # {num_subsample} "
                            f"(the raw size is {raw_size}).")
            # Modify instruction with specific language
            for sample in list_data_dict:
                sample['instruction'] = \
                    sample['category'] + ' ' + sample['instruction']
        except FileNotFoundError:
            raise FileNotFoundError(
                'Data not found! Please run `python '
                'federatedscope/llm/dataset/code_search_net.py` '
                'to download data.')
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'rosetta_alpaca':
        fp = ensure_data_file(
            'rosetta_alpaca.json',
            'https://raw.githubusercontent.com/'
            'sahil280114/codealpaca/'
            'd269da106a579a623a654529b3cb91b5dfa9c72f/'
            'data/rosetta_alpaca.json',
            description='rosetta alpaca data')
        list_data_dict = load_json(fp,
                                   instruction='instruction',
                                   input='input',
                                   output='output',
                                   category='input')

        # Remove 'x86-64 Assembl' if splitter is `meta` due to the number of
        # samples is too small.
        if config.data.splitter == 'meta':
            list_data_dict = [
                i for i in list_data_dict if i['category'] != 'X86-64 Assembly'
            ]
        # Manually remove \u00a0
        for i in range(len(list_data_dict)):
            list_data_dict[i]['output'] = \
                list_data_dict[i]['output'].replace('\u00a0', '')
            list_data_dict[i]['instruction'] = \
                list_data_dict[i]['instruction'].replace('\u00a0', '')
        dataset = LLMDataset(list_data_dict, tokenizer)

    elif dataset_name.lower() == 'offsite_tuning':
        from federatedscope.llm.dataloader.offsite_tuning_dataset import \
            PIQA, HellaSwag, OpenBookQA, ARC, SciQ, WebQs, RACE
        # list of dataset
        task_dict = {
            "piqa": PIQA(),
            "hellaswag": HellaSwag(),
            "openbookqa": OpenBookQA(),
            "arc_easy": ARC(name='ARC-Easy'),
            "arc_challenge": ARC(name='ARC-Challenge'),
            "sciq": SciQ(),
            "web_questions": WebQs(),
            "race": RACE(),
        }
        # concat these datasets
        list_train_dict, list_val_dict, list_test_dict = [], [], []
        for dataset in task_dict.values():
            list_train_dict += dataset.get_data_dict(label='train')
            list_val_dict += dataset.get_data_dict(label='validation')
            list_test_dict += dataset.get_data_dict(label='test')

        train_dataset = LLMDataset(list_train_dict,
                                   tokenizer,
                                   prompt_no_input='{context}',
                                   prompt_input='{context}',
                                   output_tag='target')
        val_dataset = LLMDataset(list_val_dict,
                                 tokenizer,
                                 prompt_no_input='{context}',
                                 prompt_input='{context}',
                                 output_tag='target')
        test_dataset = LLMDataset(list_test_dict,
                                  tokenizer,
                                  prompt_no_input='{context}',
                                  prompt_input='{context}',
                                  output_tag='target')

        dataset = (train_dataset, val_dataset, test_dataset)

    elif dataset_name.lower() == 'wikitext-2':
        pass

    else:
        raise ValueError(f'Not support data type {dataset_name}.')

    return dataset, config


if __name__ == '__main__':
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.cmd_args import parse_args, parse_client_cfg
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.auxiliaries.logging import update_logger

    init_cfg = global_cfg.clone()
    args = parse_args()
    if args.cfg_file:
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
    init_cfg.merge_from_list(cfg_opt)

    update_logger(init_cfg, clear_before_add=True)
    setup_seed(init_cfg.seed)

    load_llm_dataset(init_cfg)
