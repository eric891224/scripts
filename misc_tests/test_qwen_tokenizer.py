from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")

print(tokenizer.chat_template)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 1 + 1?"},
    {"role": "assistant", "content": "1 + 1 = 2."},
]

encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
)

print(encoded.keys())
print(encoded.get("assistant_masks"))
