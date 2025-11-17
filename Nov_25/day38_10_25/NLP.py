import nltk
nltk.download('all')
from nltk.tokenize import sent_tokenize,word_tokenize

my_data='''I am feeling happy.I wish to be good like this.
 I am blessed to have trainer like Mr.Mohan'''
s_token = sent_tokenize(my_data)
print(s_token)
w_token = word_tokenize(my_data)
print(w_token)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
my_sample = []
for word in w_token:
    if word not in stop_words:
        my_sample.append(word)
print(my_sample)




