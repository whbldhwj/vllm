from transformers import AutoConfig, PretrainedConfig

from vllm.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "mpt": MPTConfig,
    "baichuan": BaiChuanConfig,
    "aquila": AquilaConfig,
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
}

_DEFAULT_MSTAR_MODEL_REVISIONS = {
    "mstar-gpt2LMHead-1.1B-bedrock-prod": "mtl_1.1B_8K_RM_88K",
    "mstar-gpt2LMHead-6.7B-bedrock-prod-rlhf-sft": "mtl_6.7BB_sft_sft_epoch_32",
    "mstar-gpt2LMHead-26B-bedrock-prod-rlhf-sft": "mtl_26B_8K_sft_sft_epoch_32",
}

def get_config(model: str, trust_remote_code: bool) -> PretrainedConfig:
    try:
        # FIXME(Jie): Temporary hack to load M* models
        if model.startswith('mstar'):
            try:
                from mstar import AutoConfig as MStarAutoConfig
            except ImportError:
                raise RuntimeError('M* package is not installed')
            if model in _DEFAULT_MSTAR_MODEL_REVISIONS:
                model_revision = _DEFAULT_MSTAR_MODEL_REVISIONS[model]
            else:
                raise RuntimeError(f"Model revision for {model} not defined")
            mstar_config = MStarAutoConfig.from_pretrained(
                model, revision=model_revision)
            print(mstar_config)
            # Convert M* model config to GPT2 config
            setattr(mstar_config, 'architectures', ['GPT2LMHeadModel'])
            from transformers import GPT2Config
            config = GPT2Config(**mstar_config.to_dict())
        else:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code)
    except ValueError as e:
        if (not trust_remote_code and
                "requires you to execute the configuration file" in str(e)):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model)
    return config
