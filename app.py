# import the necessary modules
import io
import os
from flask import Flask, render_template, request
from google.cloud import speech_v1p1beta1 as speech
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# set the path to your Google Cloud credentials file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'endless-cabinet-380306-1322876d33e1.json'

# create a Flask app
app = Flask(__name__)


# define a route to render the HTML file
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/audio')
def audio():
    return render_template('audio.html')


@app.route('/file')
def file():
    return render_template('file.html')


# define a route to handle the form submission
@app.route('/transcribe', methods=['POST'])
def transcribe():
    # get the audio file or recorded audio from the form data
    if 'audio_file' in request.files:
        # the user uploaded an audio file
        audio_file = request.files['audio_file']
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
    else:
        # the user recorded audio
        content = request.data
        audio = speech.RecognitionAudio(content=content)

    # set up the client and the speech recognition configuration for Hindi language
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=44100,
        language_code='hi-IN',
        alternative_language_codes=['en-IN'],
        audio_channel_count=2,
        model='latest_long'
    )

    # call the Google Cloud Speech API to transcribe the audio
    response = client.recognize(config=config, audio=audio)

    # concatenate the transcribed text and word time offsets
    transcription = ''
    for result in response.results:
        alternative = result.alternatives[0]
        translated_text = GoogleTranslator(source='auto', target='en').translate(alternative.transcript)
        analyzer = SentimentIntensityAnalyzer()
        sentiment_dict = analyzer.polarity_scores(translated_text)
        if sentiment_dict['compound'] >= 0.05:
            transcription += alternative.transcript + '\n' + str(
                sentiment_dict) + '\n' + 'It is a Positive Sentence' + '\n'
        elif sentiment_dict['compound'] <= - 0.05:
            transcription += alternative.transcript + '\n' + str(
                sentiment_dict) + '\n' + 'It is a Negative Sentence' + '\n'
        else:
            transcription += alternative.transcript + '\n' + str(
                sentiment_dict) + '\n' + 'It is a Neutral Sentence' + '\n'

    # return the transcription as a response to the AJAX request
    return transcription


if __name__ == '__main__':
    app.run()
