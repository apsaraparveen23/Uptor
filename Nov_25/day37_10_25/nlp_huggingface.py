import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import pipeline

model=pipeline('text-generation',model='gpt2')
#It will do various process
my_prompt='India is'
output=model(my_prompt,max_length=70)
print('output:',output)
print(output[0]['generated_text'])
