import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    # LlamaForCausalLM,
    AutoTokenizer,
)
from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.utils import plug_quantizer_into_model
from KVcache_manager import ModelKVCacheManager
from calib_config import *

from modeling_llamagear import LlamaForCausalLM_GEARKIVI
from modeling_llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig


@torch.no_grad()
def eval_ppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset="wikitext2-v1",
    input_len: int = 2048,
):
    if dataset not in DATASET_CACHE:
        raise RuntimeError(f"{dataset} invalid")

    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")["input_ids"]

    nsamples = testenc.numel() // input_len
    nlls = []

    loss_fct = nn.CrossEntropyLoss()
    for i in tqdm(range(nsamples)):
        # [bs, input_len]
        batch = testenc[:, (i * input_len) : ((i + 1) * input_len)].to(model.device)
        outputs = model.model(batch)
        hidden_states = outputs[0]
        # [bs, input_len, vocab_size]
        logits = model.lm_head(hidden_states)
        # [bs, input_len-1, vocab_size]
        shift_logits = logits[:, :-1, :]
        # [bs, input_len-1]
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)
        loss = loss_fct(
            # [bs * (input_len-1), vocab_size]
            shift_logits.view(-1, shift_logits.size(-1)),
            # [bs * (input_len-1)]
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * input_len
        nlls.append(neg_log_likelihood)

        for layer in model.model.layers:
            manager = getattr(layer.self_attn, "KV_cache_manager", None)
            if manager is not None:
                manager.clear()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * input_len)).item()
    print(dataset, ppl)

    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")

    model_to_len = 4096

    args = parser.parse_args()
    model_name = "llama2-7b"

    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

    config.k_bits = 2# current support 2/4 bit for KV Cache
    config.v_bits = 2 # current support 2/4 bit for KV Cache
    config.group_size = 64
    config.residual_length = 64 # the number of recent fp16 tokens

    max_token = 1000

    # 0. load model and tokenizer
    compress_config = {}
    compress_config["compress_method"] = "gearlKIVI" # "gearlKIVI" "gearsKIVI"
    compress_config["group_size"] = 64
    compress_config["residual"] = 64
    compress_config["quantize_bit"] = 2
    compress_config["rank"] = 2 ## prefill rank
    compress_config["rankv"] = 2 ## prefill rank
    compress_config["loop"] = 3
    # compress_config["stream_list"] = stream_list
    stream_list = [torch.cuda.Stream(),torch.cuda.Stream()]

    args.model = "None"

    if "gearl" in args.model:
        model = LlamaForCausalLM_GEARKIVI.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            config = config,
            # quantization_config = quantization_config,
            compress_config = compress_config,
            torch_dtype=torch.float16,
            device_map = "cuda:0"
        )
    elif "KIVI" in args.model:
        model = LlamaForCausalLM_KIVI.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            config = config,
            # quantization_config = quantization_config,
            # compress_config = compress_config,
            torch_dtype=torch.float16,
            device_map = "cuda:0"
        )
    elif "None" in args.model:
        model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map = "cuda:0")
    model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-hf', 
        model_max_length=max_token,
        max_length=max_token,
        use_fast=False, 
        trust_remote_code=True, 
        tokenizer_type='llama')
    tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(model.model.layers)
    input_len = 4096

    # 1. fp16 baseline
    fp16_ppl = eval_ppl(model, tokenizer, input_len=input_len)


    # # 2. create ModelKVCacheManager
    # kv_managers_lis: list[ModelKVCacheManager] = []

    # group_set = [64]
    # for group_size in group_set:
    #     rod_meta = MODEL_TO_REORDER[model_name][group_size]["minmax"]
    #     for kbits, vbits in [
    #         (4,4), (3,3), (2,2),
    #     ]:
    #         kv_managers_lis.append(
    #             ModelKVCacheManager.create(
    #                 model,
    #                 kbits,
    #                 vbits,
    #                 group_size,
    #                 reorder_file=rod_meta,
    #                 smooth_file=None,
    #                 window_size=0,
    #                 pre_rope=True,
    #                 clipping=[0.96 for _ in range(num_layers)],
    #                 attn_sink=5,
    #                 full_prefill=False,
    #                 fp8=True,
    #                 fake_quant=True,
    #             )
    #         )

    # # 3. SKVQ PPL
    # for model_kv_manager in kv_managers_lis:
    #     model_kv_manager.full_prefill(False)
    #     plug_quantizer_into_model(model, model_kv_manager)
    #     print(model_kv_manager)
    #     ppl = eval_ppl(model, tokenizer, input_len=input_len)
