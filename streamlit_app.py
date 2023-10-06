import streamlit as st
import streamlit_ext as ste
import openai
from pydub import AudioSegment
from pytube import YouTube
import io
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.database.util import load_rttm
from pyannote.core import Annotation, Segment, notebook
import asyncio
import time
import json
import torch
import urllib.parse as urlparse
from urllib.parse import urlencode

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def create_audio_stream(audio):
    return io.BytesIO(audio.export(format="wav").read())

def add_query_parameter(link, params):
    

    url_parts = list(urlparse.urlparse(link))
    query = dict(urlparse.parse_qsl(url_parts[4]))
    query.update(params)

    url_parts[4] = urlencode(query)

    return urlparse.urlunparse(url_parts)

def youtube_video_id(value):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse.urlparse(value)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            p = urlparse.parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    # fail?
    return None

@st.cache_resource
def load_rttm_file(rttm_path):
    return load_rttm(rttm_path)['stream']


# Set your OpenAI API, Hugging Face keys
openai.api_key = st.secrets['openai'] 
hf_api_key = st.secrets['hf']

st.title("Speech Diarization and Speech-to-Text with PyAnnote and Whisper ASR")

option = st.radio("Select source:", ["Upload an audio file", "Use YouTube link","See Example"])

# Upload audio file
if option == "Upload an audio file":
    uploaded_audio = st.file_uploader("Upload an audio file (MP3 or WAV)", type=["mp3", "wav","mp4"])
    with st.expander('Optional Parameters'):
        rttm = st.file_uploader("Upload .rttm if you already have one", type=["rttm"])
        transcript_file = st.file_uploader("Upload transcipt json", type=["json"])
        youtube_link = st.text_input('Youtube link of the audio sample')

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav", start_time=0)
        audio_name = uploaded_audio.name
        audio = AudioSegment.from_file(uploaded_audio)
        
        # sample_rate = st.number_input("Enter the sample rate of the audio", min_value=8000, max_value=48000)
        # audio = audio.set_frame_rate(sample_rate)

        
# use youtube link
elif option == "Use YouTube link":
    youtube_link_raw = st.text_input("Enter the YouTube video URL:")
    youtube_link = f'https://youtu.be/{youtube_video_id(youtube_link_raw)}'
    with st.expander('Optional Parameters'):
        rttm = st.file_uploader("Upload .rttm if you already have one", type=["rttm"])
        transcript_file = st.file_uploader("Upload transcipt json", type=["json"])  
    if youtube_link_raw:
        # try:
        yt = YouTube(youtube_link)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_name = audio_stream.default_filename
        st.write(f"Fetching audio from YouTube: {youtube_link} - {audio_name}")
        audio_file = audio_stream.download(filename='sample.mp4')
        time.sleep(2)
        audio = AudioSegment.from_file('sample.mp4')
        st.audio(create_audio_stream(audio), format="audio/mp4", start_time=0)
        # sample_rate = st.number_input("Enter the sample rate of the audio", min_value=8000, max_value=48000)
        # audio = audio.set_frame_rate(sample_rate)
        # except Exception as e:
        #     st.write(f"Error: {str(e)}")
elif option == 'See Example':
    youtube_link = 'https://youtu.be/q466E9boFOY?si=A6-yt3CY_aUbYy6W'
    audio_name = 'Paul George Steve Balmer Podcast'
    st.write(f'Loading 1.5 hour audio file from {youtube_link}')
    audio = AudioSegment.from_file('example/paul-george-steve-balmer.mp4')
    rttm = "example/paul-geogre-steve-balmer-diarized.rttm"
    transcript_file = 'example/paul-george-steve-balmer-transcript.json'
    
                                    

# Diarize
if "audio" in locals():
    st.write('Performing Diarization...')
    # create stream
    duration = audio.duration_seconds
    
    
    # Perform diarization with PyAnnote
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=hf_api_key)
    if torch.cuda.device_count() > 0: # use gpu if available
        pipeline.to(torch.device('cuda'))

    # run the pipeline on an audio file
    if rttm != None:
        st.write(f'Loading {rttm}')
        diarization = load_rttm_file(rttm)
    else:
        with ProgressHook() as hook:
            audio_ = create_audio_stream(audio)
            diarization = pipeline(audio_, hook=hook)
            # dump the diarization output to disk using RTTM format
            with open(f'{audio_name.split(".")[0]}.rttm', "w") as f:
                diarization.write_rttm(f)
    
    # Display the diarization results
    st.write("Diarization Results:")
    
    annotation = Annotation()
    sp_chunks = []
    progress_text = f"Processing 1/{len(sp_chunks)}..."
    my_bar = st.progress(0, text=progress_text)
    counter = 0
    n_tracks = len([a for a in diarization.itertracks(yield_label=True)])
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        annotation[turn] = speaker
        progress_text = f"Processing {counter}/{len(sp_chunks)}..."
        my_bar.progress((counter+1)/n_tracks, text=progress_text)
        counter +=1
        temp = {'speaker': speaker,
                'start': turn.start, 'end': turn.end, 'duration': turn.end-turn.start,
                'audio': audio[turn.start*1000:turn.end*1000]}
        if transcript_file == None:
            temp['audio_stream'] = create_audio_stream(audio[turn.start*1000:turn.end*1000])
        sp_chunks.append(temp)

    # plot
    notebook.crop = Segment(-1, duration + 1)
    figure, ax = plt.subplots(figsize=(10,3))
    notebook.plot_annotation(annotation, ax=ax, time=True, legend=True)
    figure.tight_layout()
    # save to file
    st.pyplot(figure)

    st.write('Speakers and Audio Samples')
    for speaker in set(s['speaker'] for s in sp_chunks):
        temp = max(filter(lambda d: d['speaker'] == speaker, sp_chunks), key=lambda x: x['duration'])
        speak_time = sum(c['duration'] for c in filter(lambda d: d['speaker'] == speaker, sp_chunks))
        rate = 100*min((speak_time, duration))/duration
        speaker_summary  = f"{temp['speaker']} ({round(rate)}% of video duration): start={temp['start']:.1f}s stop={temp['end']:.1f}s"
        if youtube_link != None:
            speaker_summary += f" {add_query_parameter(youtube_link, {'t':str(int(temp['start']))})}"
        st.write(speaker_summary)
        st.audio(create_audio_stream(temp['audio']))


    # st.write("Transcription with Whisper ASR:")
    
    st.divider()
    # # Perform transcription with Whisper ASR
    st.write('Transcribing using Whisper...')
    container = st.container()

    progress_text = f"Processing 1/{len(sp_chunks)}..."
    my_bar = st.progress(0, text=progress_text)
    
    if  transcript_file != None:
        with open(transcript_file,'r') as f:
            sp_chunks_loaded = json.load(f)
        for i,s in enumerate(sp_chunks_loaded):
            if s['transcript'] != None:
                transcript_summary = f"{s['speaker']} start={float(s['start']):.1f}s end={float(s['end']):.1f}s: {s['transcript']}" 
                if youtube_link != None:
                    transcript_summary += f" {add_query_parameter(youtube_link, {'t':str(int(s['start']))})}"

                st.write(transcript_summary)
            progress_text = f"Processing {i+1}/{len(sp_chunks_loaded)}..."
            my_bar.progress((i+1)/len(sp_chunks_loaded), text=progress_text)

        transcript_json = sp_chunks_loaded
        transcript_path = f'example-transcript.json'

    else:
        sp_chunks_updated = []
        for i,s in enumerate(sp_chunks):
            if s['duration'] > 0.1:
                audio_path = s['audio'].export('temp.wav',format='wav')
                try:
                    transcript = openai.Audio.transcribe("whisper-1", audio_path)['text']
                except Exception:
                    transcript = ''
                    pass

                if transcript !='' and transcript != None:
                    s['transcript'] = transcript
                    transcript_summary = f"{s['speaker']} start={s['start']:.1f}s end={s['end']:.1f}s : {s['transcript']}" 
                    if youtube_link != None:
                        transcript_summary += f" {add_query_parameter(youtube_link, {'t':str(int(s['start']))})}"
                    
                    sp_chunks_updated.append({'speaker':s['speaker'], 
                                            'start':s['start'], 'end':s['end'],
                                            'duration': s['duration'],'transcript': transcript})

                    progress_text = f"Processing {i+1}/{len(sp_chunks)}..."
                    my_bar.progress((i+1)/len(sp_chunks), text=progress_text)
                    st.write(transcript_summary)

        transcript_json = [dict((k, d[k]) for k in ['speaker','start','end','duration','transcript'] if k in d) for d in sp_chunks_updated]
        transcript_path = f'{audio_name.split(".")[0]}-transcript.json'

    with open(transcript_path,'w') as f:
        json.dump(transcript_json, f)
    
    with container:
        st.info(f'Completed transcribing')
       
        @st.cache_data
        def convert_df(string):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return string.encode('utf-8')

        transcript_json_download = convert_df(json.dumps(transcript_json))

        c1_b,c2_b = st.columns((1,2))
        with c1_b:
            ste.download_button(
                "Download transcript as json",
                transcript_json_download,
                transcript_path,
            )

        header = ','.join(transcript_json[0].keys()) + '\n'
        for s in transcript_json:
            header += ','.join([str(e) if ',' not in str(e) else '"' + str(e) + '"' for e in s.values()]) + '\n'

        transcript_csv_download = convert_df(header)
        with c2_b:
            ste.download_button(
                "Download transcript as csv",
                transcript_csv_download,
                f'{audio_name.split(".")[0]}-transcript.csv'
            )
    
      
    

    
    # st.write(transcription_result['text'])
