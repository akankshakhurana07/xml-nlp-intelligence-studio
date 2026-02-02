# ================= PAGE CONFIG =================
import streamlit as st
st.set_page_config(
    page_title="XML NLP Intelligence Studio",
    page_icon="🧠",
    layout="wide"
)

# ================= UI / CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at top, #020617, #000000);
    color: #e5e7eb;
}

/* HERO */
.hero {
    font-size: 3.4rem;
    font-weight: 900;
    background: linear-gradient(90deg,#38bdf8,#22d3ee,#818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 2.8s ease-in-out infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 10px #38bdf8; }
    to   { text-shadow: 0 0 35px #818cf8; }
}

/* SUBTITLE */
.subtitle {
    font-size: 1.15rem;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* CARDS */
.card {
    background: rgba(2,6,23,.82);
    backdrop-filter: blur(16px);
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 28px;
    border: 1px solid rgba(56,189,248,.25);
    box-shadow: 0 0 40px rgba(56,189,248,.12);
    transition: all .35s ease;
}
.card:hover {
    transform: translateY(-6px);
    box-shadow: 0 0 55px rgba(56,189,248,.35);
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#000000);
    border-right: 1px solid #1e293b;
}

/* DOWNLOAD BUTTONS */
button[kind="primary"] {
    background: linear-gradient(90deg,#38bdf8,#818cf8);
    color: black;
    border-radius: 14px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ================= IMPORTS =================
import re
import xml.etree.ElementTree as ET
import nltk
import matplotlib.pyplot as plt
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.util import ngrams

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from wordcloud import WordCloud

# ================= NLTK DOWNLOADS =================
for pkg in [
    "punkt",
    "stopwords",
    "averaged_perceptron_tagger",
    "wordnet",
    "maxent_ne_chunker",
    "words"
]:
    nltk.download(pkg)

# ================= HEADER =================
st.markdown('<div class="hero">🧠 XML NLP Intelligence Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Clean • Stable • Industry-Style NLP Pipeline on XML Data</div>',
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload XML File", type=["xml"])
show_raw = st.sidebar.checkbox("Show Extracted Text", True)
show_wc = st.sidebar.checkbox("Show WordCloud", True)

# ================= MAIN LOGIC =================
if uploaded_file:

    # ---------- XML CLEANING ----------
    tree = ET.parse(uploaded_file)
    root = tree.getroot()
    text = " ".join(e.text.strip() for e in root.iter() if e.text)
    text = re.sub(r"\s+", " ", text)

    # ---------- TOKENIZATION ----------
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # ---------- STOPWORDS ----------
    stop_words = set(stopwords.words("english"))
    clean_words = [w for w in words if w.isalpha() and w not in stop_words]

    # ---------- LEMMATIZATION ----------
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in clean_words]

    # ---------- POS & NER ----------
    pos_tags = pos_tag(lemmas)
    ner_tree = ne_chunk(pos_tags)

    # ---------- BIGRAMS ----------
    bigrams = list(ngrams(lemmas, 2))

    # ---------- METRICS ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentences", len(sentences))
    c2.metric("Tokens", len(words))
    c3.metric("Clean Tokens", len(lemmas))
    c4.metric("Vocabulary", len(set(lemmas)))

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧹 Text Cleaning",
        "🧠 Linguistic Analysis",
        "📊 Vectorization",
        "☁️ Visualization",
        "⬇️ Downloads"
    ])

    # ---------- TAB 1 ----------
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if show_raw:
            st.write(text[:2200])
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- TAB 2 ----------
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("POS Tagging")
        st.write(pos_tags[:20])

        st.subheader("Named Entity Recognition")
        st.write(ner_tree)

        st.subheader("Bigrams")
        st.write(bigrams[:15])
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- TAB 3 ----------
    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        bow = CountVectorizer()
        bow.fit([" ".join(lemmas)])
        st.write("📌 Bag of Words Vocabulary Size:", len(bow.get_feature_names_out()))

        tfidf = TfidfVectorizer()
        tfidf.fit([" ".join(lemmas)])
        st.write("📌 TF-IDF Feature Size:", len(tfidf.get_feature_names_out()))

        w2v = Word2Vec([lemmas], vector_size=100, window=5, min_count=1)
        st.write("📌 Word2Vec Vector Sample:", w2v.wv[lemmas[0]][:10])

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- TAB 4 ----------
    with tab4:
        if show_wc:
            wc = WordCloud(
                width=900,
                height=420,
                background_color="black",
                colormap="cool"
            ).generate(" ".join(lemmas))
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

    # ---------- TAB 5 : DOWNLOADS ----------
    with tab5:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.download_button(
            "📄 Download Cleaned Text (TXT)",
            data=text,
            file_name="cleaned_text.txt",
            mime="text/plain"
        )

        lemmas_df = pd.DataFrame({"lemmas": lemmas})
        st.download_button(
            "📑 Download Tokens / Lemmas (CSV)",
            data=lemmas_df.to_csv(index=False),
            file_name="tokens_lemmas.csv",
            mime="text/csv"
        )

        bow_df = pd.DataFrame(
            bow.transform([" ".join(lemmas)]).toarray(),
            columns=bow.get_feature_names_out()
        )
        st.download_button(
            "📊 Download BoW Features (CSV)",
            data=bow_df.to_csv(index=False),
            file_name="bow_features.csv",
            mime="text/csv"
        )

        tfidf_df = pd.DataFrame(
            tfidf.transform([" ".join(lemmas)]).toarray(),
            columns=tfidf.get_feature_names_out()
        )
        st.download_button(
            "📈 Download TF-IDF Features (CSV)",
            data=tfidf_df.to_csv(index=False),
            file_name="tfidf_features.csv",
            mime="text/csv"
        )

        sample_words = list(dict.fromkeys(lemmas))[:10]
        w2v_df = pd.DataFrame(
            {w: w2v.wv[w][:10] for w in sample_words}
        ).T
        st.download_button(
            "🧠 Download Word2Vec Sample (CSV)",
            data=w2v_df.to_csv(),
            file_name="word2vec_sample.csv",
            mime="text/csv"
        )

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("⬅️ Upload an XML file from the sidebar to begin NLP processing")
