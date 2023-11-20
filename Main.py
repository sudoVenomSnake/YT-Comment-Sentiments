from youtube_comment_downloader import *
from itertools import islice
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from stqdm import stqdm
from pytube import YouTube
import numpy as np
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(layout = "wide", page_title = "YouTube Sentiment Analysis")

st.title("YouTube Sentiment Analysis")

if "initial_submit" not in st.session_state:
    st.session_state.initial_submit = False

def initial_submit():
    st.session_state.initial_submit = True

@st.cache_resource
def initialize():
    downloader = YoutubeCommentDownloader()
    analyzer = SentimentIntensityAnalyzer()
    return downloader, analyzer

@st.cache_resource
def get_comments(url, type, limit):
    comments_extracted = []
    comments = downloader.get_comments_from_url(url, sort_by = type)
    for comment in stqdm(islice(comments, limit)):
        comments_extracted.append(comment)
    return comments_extracted

downloader, analyzer = initialize()

with st.form("initial-submit"):
    url = st.text_input(label = "Please input a valid YouTube URL", value = "https://www.youtube.com/watch?v=gQddtTdmG_8")
    num_comments = st.slider(label = "Number of comments to analyze", min_value = 10, max_value = 1000, value = 335)
    submit = st.form_submit_button(label = "Analyse", on_click = initial_submit)

if st.session_state.initial_submit:
    comments = get_comments(url, SORT_BY_POPULAR, num_comments)
    text = []
    top_3 = []
    p_sentiment = []
    n_sentiment = []
    neu_sentiment = []
    c = 0
    for i in comments:
        if c != 5:
            c += 1
            temp = {}
            temp["text"] = i["text"]
            temp["time"] = i["time"]
            temp["author"] = i["author"]
            temp["votes"] = i["votes"]
            temp["photo"] = i["photo"]
            top_3.append(temp)
        text.append(i["text"])
        sentiments = analyzer.polarity_scores(i["text"])
        p_sentiment.append(sentiments["pos"])
        n_sentiment.append(sentiments["neg"])
        neu_sentiment.append(sentiments["neu"])

    yt = YouTube(url)
    st.header(yt.title)
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.divider()
        col_1, col_2 = st.columns(2)
        with col_1:
            st.metric(label = "Avg. Positive Sentiment", value = round(np.mean(p_sentiment), 2))
        with col_2:
            st.metric(label = "Avg. Negative Sentiment", value = round(np.mean(n_sentiment), 2))
        st.metric(label = "Avg. Neutral Sentiment", value = round(np.mean(neu_sentiment), 2))
        st.markdown(f"### Total Length : **{yt.length}** Seconds")
        st.markdown(f"### Total Views : **{yt.views}**")
        st.markdown(f"### Is Age Restricted : {yt.age_restricted}")
    with col2:
        st.image(yt.thumbnail_url)
        
    
    st.header("Top Comments -")
    for i in range(3):
        col1, col2, col3, col4, col5 = st.columns([1, 5, 1, 1, 1], gap = "small")
        with col1:
            st.image(top_3[i]["photo"])
        with col2:
            with st.container():
                st.markdown(f"**{top_3[i]['author']}**")
                st.text(top_3[i]["text"])
        with col3:
            st.metric(label = "Positive Sentiment", value = round(p_sentiment[i], 2))
        with col4:
            st.metric(label = "Negative Sentiment", value = round(n_sentiment[i], 2))
        with col5:
            st.metric(label = "Neutral Sentiment", value = round(neu_sentiment[i], 2))

    comments = get_comments(url, SORT_BY_RECENT, num_comments)
    text = []
    p_sentiment = []
    n_sentiment = []
    neu_sentiment = []
    val = []
    color = []
    for i in comments:
        text.append(i["text"])
        sentiments = analyzer.polarity_scores(i["text"])
        p_sentiment.append(sentiments["pos"])
        n_sentiment.append(sentiments["neg"])
        neu_sentiment.append(sentiments["neu"])
        value = sentiments["pos"] - sentiments["neg"]
        val.append(value)
        color.append("mediumslateblue" if value > 0 else "tomato")
    
    # st.write(color)
    # st.plotly_chart(px.bar(y = val, color_discrete_sequence = color), use_container_width = True)
    colors = ['crimson' if x < 0 else 'lightseagreen' for x in val]

    st.header("Sentiment Time Series -")
    fig = go.Figure(data = [go.Bar(
        x = list(range(len(val))),
        y = val,
        marker_color = colors
    )])
    st.plotly_chart(fig, use_container_width = True)

    st.header("Sentiment DataFrame -")
    df = pd.DataFrame({"Comments" : text, "Positive Sentiment" : p_sentiment, "Negative Sentiment" : n_sentiment, "Neutral Sentiment" : neu_sentiment})
    st.dataframe(df)
    st.download_button("Download Analysis", df.to_csv(index = False).encode('utf-8'), "file.csv", "text/csv", key = 'download-csv')
