import os, torch
from tokenizers import Tokenizer, models, pre_tokenizers, Regex
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
vocab = {str(i): i for i in range(10)}
vocab.update({"\n": 10, "user": 11, "assistant": 12, "<|endoftext|>": 13, "<|im_start|>": 14, "<|im_end|>": 15})
tk = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>"))
tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"\d|\n|[a-z]+"), behavior="isolated")
tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<|endoftext|>", eos_token="<|im_end|>",
                              additional_special_tokens=["<|im_start|>", "<|im_end|>"])
cfg = Qwen3Config(vocab_size=16, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, max_position_embeddings=8192, tie_word_embeddings=False)
torch.manual_seed(0)
m = Qwen3ForCausalLM(cfg)
out = "tiny_model"; os.makedirs(out, exist_ok=True)
m.save_pretrained(out); tok.save_pretrained(out)
t2 = PreTrainedTokenizerFast.from_pretrained(out)
print(t2.encode("<|im_start|>user\n01\n23<|im_end|><|im_start|>assistant\n", add_special_tokens=False))
