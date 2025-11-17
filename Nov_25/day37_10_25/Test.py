import sounddevice as sd
import queue
import json
import os
import numpy as np
from vosk import Model, KaldiRecognizer
from gTTS import gTTS
import playsound
import pandas as pd


class TalkBot:
    LANGUAGE_EN = 'en'
    question = None

    def __init__(self):
        print("Loading Vosk model...")
        self.model = Model("vosk-model-small-en-us-0.15")
        self.samplerate = 16000
        self.q = queue.Queue()

    def callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def speechToText(self):
        print("Devotee, please tell us your problem...")

        with sd.RawInputStream(samplerate=self.samplerate, blocksize=8000,
                               dtype='int16', channels=1, callback=self.callback):
            recognizer = KaldiRecognizer(self.model, self.samplerate)

            while True:
                data = self.q.get()
                if recognizer.AcceptWaveform(data):
                    result = recognizer.Result()
                    text = json.loads(result)["text"]
                    self.question = text
                    print("You said:", text)
                    self.getAnswer()
                    break

    def textToSpeech(self, answer):
        audio_file = "temp.mp3"
        tts = gTTS(text=answer, lang=self.LANGUAGE_EN)
        tts.save(audio_file)
        playsound.playsound(audio_file)

    def getAnswer(self):
        try:
            data = pd.read_csv("sai_baba_question_answer.csv")

            # simple match: pick random answer for now
            answer = data["answer"].sample(1).iloc[0]

            print("Answer:", answer)
            self.textToSpeech(answer)

        except Exception as e:
            print("Error:", e)

def main():
    bot = TalkBot()
    while True:
        bot.speechToText()
        cont = input("Ask another question? (yes/no): ")
        if cont.lower() != "yes":
            break


if __name__ == "__main__":
    main()
