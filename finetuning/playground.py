from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

device = 1
tokenizer = T5Tokenizer.from_pretrained("t5-base")
# T5_config = T5Config().from_pretrained("t5-base")
# model = T5ForConditionalGeneration(config=T5_config)
model = T5ForConditionalGeneration.from_pretrained("t5-base")

model = model.to(device)
ids = "I enjoy walking with my cute dog"
ids = tokenizer(ids, return_tensors="pt")["input_ids"].to(device)
# print(ids)
outputs = model.generate(
    ids,
    max_length=50,
    num_beams=5,
    num_return_sequences=5,
    no_repeat_ngram_size=5,
    # output_scores=True,
    # return_dict_in_generate=False,
    early_stopping=True,
)
# print(outputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# print(tokenizer.decode(outputs[1], skip_special_tokens=True))
outputs = model.generate(ids, max_length=50, do_sample=True, top_k=50, temperature=0.7)
# print(outputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

outputs = model.generate(ids, num_beams=5, max_length=50, do_sample=True, top_p=0.92)
# print(outputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
