import os

import torch

from federatedscope.llm.model.adapter_builder import AdapterModel


def parse_model_type(model_type, default_hub='huggingface_llm'):
    if '@' in model_type:
        return model_type.rsplit('@', 1)
    return model_type, default_hub


def get_model_cache_dir(config):
    cache_dir = getattr(config.llm.cache, 'model', '')
    return cache_dir if len(cache_dir) else None


def prepare_pretrained_kwargs(model_name, config, **kwargs):
    model_name = os.path.expanduser(model_name)
    if os.path.exists(model_name):
        kwargs.setdefault('local_files_only', True)
    else:
        cache_dir = get_model_cache_dir(config)
        if cache_dir is not None:
            kwargs.setdefault('cache_dir', cache_dir)

    if config.train.is_enable_half:
        kwargs.setdefault('torch_dtype', torch.bfloat16)

    return model_name, kwargs


def get_model_from_huggingface(model_name, config, **kwargs):
    from transformers import AutoModelForCausalLM

    model_name, kwargs = prepare_pretrained_kwargs(model_name, config,
                                                   **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_model_from_modelscope(model_name, config, **kwargs):
    from modelscope import AutoModelForCausalLM

    model_name, kwargs = prepare_pretrained_kwargs(model_name, config,
                                                   **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def get_llm(config, **kwargs):
    from federatedscope.llm.dataloader import get_tokenizer

    model_config = config.model
    model_name, model_hub = parse_model_type(model_config.type)
    if model_hub == 'huggingface_llm':
        model = get_model_from_huggingface(model_name=model_name,
                                           config=config,
                                           **kwargs)
    elif model_hub == 'modelscope_llm':
        model = get_model_from_modelscope(model_name=model_name,
                                          config=config,
                                          **kwargs)
    else:
        raise NotImplementedError(f'Not support LLM {model_name} in'
                                  f' {model_hub}.')

    # Resize LLM model based on settings
    tokenizer, num_new_tokens = \
        get_tokenizer(model_name, get_model_cache_dir(config),
                      config.llm.tok_len)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    args = config.llm.adapter.args[0] if len(
        config.llm.adapter.args[0]) > 0 else {}
    model = AdapterModel(model, use_adapter=config.llm.adapter.use, **args)

    return model
