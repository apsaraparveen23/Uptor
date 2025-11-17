from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Pick a suitable checkpoint — e.g. one of:
# "gopi30/english-to-tamil-stage1" :contentReference[oaicite:0]{index=0}
# or "suriya7/English-to-Tamil" :contentReference[oaicite:1]{index=1}
# or "Mr-Vicky-01/English-Tamil-Translator" :contentReference[oaicite:2]{index=2}

checkpoint = "gopi30/english-to-tamil-stage1"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

def translate_en_to_ta(text: str, max_len: int = 128):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    # Generate translated output
    outputs = model.generate(**inputs, max_length=max_len)
    # Decode and clean up
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    en_text = "Machine learning is amazing!"
    ta_translation = translate_en_to_ta(en_text, max_len=40)
    print("Tamil:", ta_translation)
