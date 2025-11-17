import speech_recognition as sr
from gtts import gTTS
import os
import pandas as pd
import numpy as np
import torch
from transformers import pipeline

class TalkBot:
    LANGUAGE_EN = 'en'
    question = None

    def speechToText(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Devotee, Please tell us your problem: ")
            audio_text = r.listen(source)
            print("Loading ...")
            try:
                self.question = r.recognize_google(audio_text)
                self.getAnswer()
            except Exception as e:
                print("Sorry, I did not get that. Please try again.")

    def textToSpeech(self, answer):
        myobj = gTTS(text=answer, lang=self.LANGUAGE_EN, slow=False)
        audio_file = "temp_audio.mp3"
        myobj.save(audio_file)
        os.system(f"mpg321 {audio_file}")  # Use f-string for clarity

    def getAnswer(self):
        try:
            data = pd.read_csv("sai_baba_question_answer.csv")
            random_num = np.random.randint(0, len(data))
            context = data["answer"][random_num]

            question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad', device=0)
            result = question_answerer(question=self.question, context=context)
            answer = result['answer']
            print("Answer:", answer)  # Print the answer in the console
            self.textToSpeech(answer)  # Convert answer to speech
        except Exception as e:
            print(e)

def main():
    talkBot = TalkBot()
    while True:
        talkBot.speechToText()
        continue_prompt = input("Would you like to ask another question? (yes/no): ").strip().lower()
        if continue_prompt != 'yes':
            print("Thank you for using TalkBot. Goodbye!")
            break

if __name__ == "__main__":
    main()