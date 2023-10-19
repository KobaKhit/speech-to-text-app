import streamlit as st
import streamlit_ext as ste
import openai
from pydub import AudioSegment
from pytube import YouTube
import pytube
import io
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.database.util import load_rttm
from pyannote.core import Annotation, Segment, notebook
import time
import json
import torch
import urllib.parse as urlparse
from urllib.parse import urlencode
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

st.set_page_config(
        page_title="Speech-to-chat",
        page_icon = '🌊',
        layout='wide'
)

# Set your OpenAI, Hugging Face API keys
openai.api_key = st.secrets['openai'] 
hf_api_key = st.secrets['hf']

TRANSCRIPTION_REQUEST_LIMIT = 150
PROMPT_REQUEST_LIMIT = 2

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

@st.cache_data
def process_youtube_link(youtube_link):
    st.write(f"Fetching audio from YouTube: {youtube_link}")
    try:
        yt = YouTube(youtube_link)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_name = audio_stream.default_filename
        st.write(f"Downloaded {audio_name}")
    except pytube.exceptions.AgeRestrictedError:
        st.stop('Age restricted videos cannot be processed.')

    try:
        os.remove('sample.mp4')
    except OSError:
        pass
    audio_file = audio_stream.download(filename='sample.mp4')
    time.sleep(2)
    audio = load_audio('sample.mp4')
    st.audio(create_audio_stream(audio), format="audio/mp4", start_time=0)
    return audio, audio_name

@st.cache_data
def load_rttm_file(rttm_path):
    return load_rttm(rttm_path)['stream']

@st.cache_resource
def load_audio(uploaded_audio):
    return AudioSegment.from_file(uploaded_audio)


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "prompt_request_counter" not in st.session_state:
    st.session_state["prompt_request_counter"] = 0

initial_prompt =  [{"role": "system", "content": "You are helping to analyze and summarize a transcript of a conversation."},
                   {"role": 'user', "content": 'Please summarize briefly the following transcript\n{}'}]
if "messages" not in st.session_state:
    st.session_state.messages = initial_prompt 
    

st.title("Speech to Chat")
reddit_thread = 'https://www.reddit.com/r/dataisbeautiful/comments/17413bq/oc_speech_diarization_app_that_transcribes_audio'

with st.sidebar:
    st.markdown('''
    # How to Use

      1. Enter a youtube link or upload an audio file.
      2. "Chat" with the file.

      Example prompts:
      - Which speaker spoke the most?
      - What are important keywords in the transcript for SEO?
    ''')

    st.divider()

    st.markdown(f'''
        # About 

        Given an audio file or a youtube link this app will
          - [x] 1. Parition the audio according to the identity of each speaker (diarization) using `pyannote` [HuggingFace Speaker Diarization api](https://huggingface.co/pyannote/speaker-diarization-3.0)
          - [x] 2. Transcribe each audio segment using [OpenAi Whisper API](https://platform.openai.com/docs/guides/speech-to-text/quickstart)
          - [x] 3. Set up an LLM chat with the transcript loaded into its knowledge database, so that a user can "talk" to the transcript of the audio file.

        This version will only process up to first 6 minutes of an audio file due to limited resources of free tier Streamlit.io/HuggingFace Spaces.
        A local version with access to a GPU can process 1 hour of audio in 1 to 5 minutes.
        If you would like to use this app at scale reach out directly by creating an issue on [github🤖](https://github.com/KobaKhit/speech-to-text-app/issues)!
        
        Rule of thumb, for this free tier hosted app it takes half the duration of the audio to complete processing, ex. g. 6 minute youtube video will take 3 minutes to diarize.

        Made by [kobakhit](https://github.com/KobaKhit/speech-to-text-app)
    ''')


# Chat container
container_transcript_chat = st.container()

# Source Selection
option = st.radio("Select source:", ["Upload an audio file", "Use YouTube link","Example"], index=2)

# Upload audio file
if option == "Upload an audio file":
    with st.form('uploaded-file', clear_on_submit=True):
        uploaded_audio = st.file_uploader("Upload an audio file (MP3 or WAV)", type=["mp3", "wav","mp4"])
        st.form_submit_button()
        if st.form_submit_button(): st.session_state.messages = initial_prompt
    with st.expander('Optional Parameters'):
        # st.session_state.rttm  = st.file_uploader("Upload .rttm if you already have one", type=["rttm"])
        # st.session_state.transcript_file = st.file_uploader("Upload transcipt json", type=["json"])
        youtube_link = st.text_input('Youtube link of the audio sample')

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav", start_time=0)
        audio_name = uploaded_audio.name
        audio = load_audio(uploaded_audio)
        
        # sample_rate = st.number_input("Enter the sample rate of the audio", min_value=8000, max_value=48000)
        # audio = audio.set_frame_rate(sample_rate)
        
# use youtube link
elif option == "Use YouTube link":        

    with st.form('youtube-link', clear_on_submit=True):
        youtube_link_raw = st.text_input("Enter the YouTube video URL:")
        youtube_link = f'https://youtu.be/{youtube_video_id(youtube_link_raw)}'
        
        if st.form_submit_button(): # reset variables on new link submit
            st.session_state.messages = initial_prompt
            st.session_state.rttm = None
            st.session_state.transcript_file = None
            st.session_state.prompt_request_counter = 0

    # with st.expander('Optional Parameters'):
    #     st.session_state.rttm = st.file_uploader("Upload .rttm if you already have one", type=["rttm"])
    #     st.session_state.transcript_file = st.file_uploader("Upload transcipt json", type=["json"])  
    if youtube_link_raw:
        audio, audio_name = process_youtube_link(youtube_link)
        # sample_rate = st.number_input("Enter the sample rate of the audio", min_value=8000, max_value=48000)
        # audio = audio.set_frame_rate(sample_rate)
        # except Exception as e:
        #     st.write(f"Error: {str(e)}")
elif option == 'Example':
    youtube_link = 'https://www.youtube.com/watch?v=TamrOZX9bu8'
    audio_name = 'Stephen A. Smith has JOKES with Shannon Sharpe'
    st.write(f'Loaded audio file from {youtube_link} - {audio_name} 👏😂')
    if os.path.isfile('example/steve a smith jokes.mp4'):
        audio = load_audio('example/steve a smith jokes.mp4')
    else:
        yt = YouTube(youtube_link)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file = audio_stream.download(filename='sample.mp4')
        time.sleep(2)
        audio = load_audio('sample.mp4')

    if os.path.isfile("example/steve a smith jokes.rttm"):
        st.session_state.rttm = "example/steve a smith jokes.rttm"
    if os.path.isfile('example/steve a smith jokes.json'):
        st.session_state.transcript_file = 'example/steve a smith jokes.json'

    st.audio(create_audio_stream(audio), format="audio/mp4", start_time=0)

# Diarize
if "audio" in locals():
    # create stream
    duration = audio.duration_seconds
    if duration > 360:
        st.info('Only processing the first 6 minutes of the audio due to Streamlit.io resource limits.')
        audio = audio[:360*1000]
        duration = audio.duration_seconds
    
    
    # Perform diarization with PyAnnote
    pipeline = Pipeline.from_pretrained(
       "pyannote/speaker-diarization-3.0", use_auth_token=hf_api_key)
    if torch.cuda.device_count() > 0: # use gpu if available
        pipeline.to(torch.device('cuda'))

    # run the pipeline on an audio file
    with st.spinner('Performing Diarization...'):
        if 'rttm' in st.session_state and st.session_state.rttm != None:
            st.write(f'Loading {st.session_state.rttm}')
            diarization = load_rttm_file(st.session_state.rttm )
        else:
            # with ProgressHook() as hook:
            audio_ = create_audio_stream(audio)
            # diarization = pipeline(audio_, hook=hook)
            diarization = pipeline(audio_)
            # dump the diarization output to disk using RTTM format
            with open(f'{audio_name.split(".")[0]}.rttm', "w") as f:
                diarization.write_rttm(f)
            st.session_state.rttm = f'{audio_name.split(".")[0]}.rttm'
    
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
        if 'transcript_file' in st.session_state and st.session_state.transcript_file == None:
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
    with st.expander('Samples', expanded=True):
        for speaker in set(s['speaker'] for s in sp_chunks):
            temp = max(filter(lambda d: d['speaker'] == speaker, sp_chunks), key=lambda x: x['duration'])
            speak_time = sum(c['duration'] for c in filter(lambda d: d['speaker'] == speaker, sp_chunks))
            rate = 100*min((speak_time, duration))/duration
            speaker_summary  = f"{temp['speaker']} ({round(rate)}% of video duration): start={temp['start']:.1f}s stop={temp['end']:.1f}s"
            if youtube_link != None:
                speaker_summary += f" {add_query_parameter(youtube_link, {'t':str(int(temp['start']))})}"
            st.write(speaker_summary)
            st.audio(create_audio_stream(temp['audio']))
    
    st.divider()
    # # Perform transcription with Whisper ASR

    
    # Transcript containers
    st.write('Transcribing using Whisper API (150 requests limit)...')
    container_transcript_completed = st.container()

    progress_text = f"Processing 1/{len(sp_chunks[:TRANSCRIPTION_REQUEST_LIMIT])}..."
    my_bar = st.progress(0, text=progress_text)
    # rework the loop. Simplify if Else
    with st.expander('Transcript', expanded=True):
        if 'transcript_file' in st.session_state and st.session_state.transcript_file != None:
            with open(st.session_state.transcript_file,'r') as f:
                sp_chunks_loaded = json.load(f)
            for i,s in enumerate(sp_chunks_loaded):
                if s['transcript'] != None:
                    transcript_summary = f"**{s['speaker']}** start={float(s['start']):.1f}s end={float(s['end']):.1f}s: {s['transcript']}" 
                    if youtube_link != None and youtube_link != '':
                        transcript_summary += f" {add_query_parameter(youtube_link, {'t':str(int(s['start']))})}"

                    st.markdown(transcript_summary)
                progress_text = f"Processing {i+1}/{len(sp_chunks_loaded)}..."
                my_bar.progress((i+1)/len(sp_chunks_loaded), text=progress_text)

            transcript_json = sp_chunks_loaded
            transcript_path = f'{audio_name.split(".mp4")[0]}-transcript.json'

        else:
            sp_chunks_updated = []
            for i,s in enumerate(sp_chunks[:TRANSCRIPTION_REQUEST_LIMIT]):
                if s['duration'] > 0.1:
                    audio_path = s['audio'].export('temp.wav',format='wav')
                    try:
                        transcript = openai.Audio.transcribe("whisper-1", audio_path)['text']
                    except Exception:
                        transcript = ''
                        pass

                    if transcript !='' and transcript != None:
                        s['transcript'] = transcript
                        transcript_summary = f"**{s['speaker']}** start={s['start']:.1f}s end={s['end']:.1f}s : {s['transcript']}" 
                        if youtube_link != None:
                            transcript_summary += f" {add_query_parameter(youtube_link, {'t':str(int(s['start']))})}"
                        
                        sp_chunks_updated.append({'speaker':s['speaker'], 
                                                'start':s['start'], 'end':s['end'],
                                                'duration': s['duration'],'transcript': transcript})

                        progress_text = f"Processing {i+1}/{len(sp_chunks[:TRANSCRIPTION_REQUEST_LIMIT])}..."
                        my_bar.progress((i+1)/len(sp_chunks[:TRANSCRIPTION_REQUEST_LIMIT]), text=progress_text)
                        st.markdown(transcript_summary)

            transcript_json = [dict((k, d[k]) for k in ['speaker','start','end','duration','transcript'] if k in d) for d in sp_chunks_updated]
            transcript_path = f'{audio_name.split(".mp4")[0]}-transcript.json'
            st.session_state.transcript_file = transcript_path

    # save the trancript file
    with open(transcript_path,'w') as f:
        json.dump(transcript_json, f)

    # generate transcript string
    transcript_string = '\n'.join([f"{s['speaker']} start={s['start']:.1f}s end={s['end']:.1f}s : {s['transcript']}" for s in transcript_json])
    
    @st.cache_data
    def get_initial_response(transcript):
        st.session_state.messages[1]['content'] = st.session_state.messages[1]['content'].format(transcript)
        initial_response = openai.ChatCompletion.create(
                            model=st.session_state["openai_model"],
                            messages=st.session_state.messages
                        )
        return initial_response['choices'][0]['message']['content']
    
    # Chat container
    with container_transcript_chat:
        # get a summary of transcript from ChatGpt
        init = get_initial_response(transcript_string)
        # pass transcript to initial prompt
        st.session_state.messages[1]['content'] = st.session_state.messages[1]['content'].format(transcript_string)
        
        # LLM Chat
        with st.expander('Summary of the Transcribed Audio File Generated by ChatGPT', expanded = True):
            # display the AI generated summary.
            with st.chat_message("assistant", avatar='https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg'):
                st.write(init)

            # chat field
            with st.form("Chat",clear_on_submit=True):
                prompt = st.text_input('Chat with the Transcript (2 prompts limit)')
                st.form_submit_button()
            
            # message list
            # for message in st.session_state.messages[2:]:
            #     with st.chat_message(message["role"]):
            #         st.markdown(message["content"])

            # make request if prompt was entered
            if prompt:
                st.session_state.prompt_request_counter += 1
                if st.session_state.prompt_request_counter > PROMPT_REQUEST_LIMIT:
                    st.warning('Exceeded prompt limit.'); 
                    st.stop()
                # append user prompt to messages
                st.session_state.messages.append({"role": "user", "content": prompt})

                # dislay user prompt
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # stream LLM Assisstant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""

                    # stream response
                    for response in openai.ChatCompletion.create(
                        model=st.session_state["openai_model"],
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                    ):
                        full_response += response.choices[0].delta.get("content", "")
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                # append ai response to messages
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            
    # Trancription Completed Section
    with container_transcript_completed:
        st.info(f'Completed transcribing')
       
        @st.cache_data
        def convert_df(string):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return string.encode('utf-8')
        # encode transcript string
        transcript_json_download = convert_df(json.dumps(transcript_json))
        # transcript download buttons
        c1_b,c2_b = st.columns((1,2))

        # json button
        with c1_b:
            ste.download_button(
                "Download transcript as json",
                transcript_json_download,
                transcript_path,
            )

        # create csv string
        header = ','.join(transcript_json[0].keys()) + '\n'
        for s in transcript_json:
            header += ','.join([str(e) if ',' not in str(e) else '"' + str(e) + '"' for e in s.values()]) + '\n'

        # csv button
        transcript_csv_download = convert_df(header)
        with c2_b:
            ste.download_button(
                "Download transcript as csv",
                transcript_csv_download,
                f'{audio_name.split(".")[0]}-transcript.csv'
            )
    