from googletrans import Translator


def korean_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text
