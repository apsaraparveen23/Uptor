from transformers import pipeline

# Load pretrained sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test it
result = classifier("I love KHOLI but i hate RCB")
print(result)
