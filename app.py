import io
import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
import speech_recognition as sr

from nlp_engine import ReviewAssistant, load_reviews, save_correction


st.set_page_config(
    page_title="Rojak | Product Enquiry Assistant",
    page_icon="🛍️",
    layout="wide",
)

DATA_PATH = Path("data/dataset.csv")


@st.cache_data(show_spinner=False)
def load_data(path_string, modified):
    return load_reviews(path_string)


@st.cache_resource(show_spinner=False)
def build_assistant(df):
    return ReviewAssistant(df)


def transcribe_recording(audio_bytes, language):
    recognizer = sr.Recognizer()

    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        recorded = recognizer.record(source)

    return recognizer.recognize_google(
        recorded,
        language=language
    )


st.markdown(
    """<style>
    .block-container{max-width:1100px;padding-top:2rem}
    .hero{padding:1.7rem;border-radius:20px;background:linear-gradient(120deg,#172554,#334155);color:white;margin-bottom:1rem}
    .hero h1{font-size:2.35rem;margin:0}.hero p{opacity:.88;margin:.4rem 0 0}
    .product-card{border:1px solid #e5e7eb;border-radius:18px;padding:1.2rem;background:rgba(128,128,128,.04)}
    .small-muted{color:#8b8b8b;font-size:.88rem}
    [data-testid="stChatMessage"]{border-radius:14px}
    </style>""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>🛍️ Rojak</h1>'
    '<p>Your Malaysian Rojak product enquiry assistant · '
    'Ask in English, Bahasa Melayu or mixed language.</p></div>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Please place Pheiying's dataset.csv inside the data/ folder.")
    st.markdown(
        "Download it from the [original GitHub dataset]"
        "(https://github.com/pheiying/Sentiment-Analysis-on-Out-Of-Vocabulary-OOV-Malaysia-Rojak-Language/blob/main/Dataset/dataset.csv), "
        "then save it as `data/dataset.csv`."
    )
    st.code("rojak-assistant/\n  app.py\n  nlp_engine.py\n  data/\n    dataset.csv")
    st.stop()

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not read dataset.csv: {exc}")
    st.stop()

if df.empty:
    st.error("No usable reviews were found. Check that the CSV contains a Review column.")
    st.stop()

assistant = build_assistant(df)

if "selected_product" not in st.session_state:
    st.session_state.selected_product = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None


with st.sidebar:
    st.title("Rojak workspace")
    st.caption("Dataset-powered product enquiry")
    page = st.radio(
        "Navigate",
        ["Product & Chat", "Review Insights", "NLP Lab & Syllabus", "Feedback & Learning"],
        label_visibility="collapsed",
    )
    st.divider()
    st.metric("Reviews loaded", len(df))
    st.caption(
        "Product coverage follows the dataset. This initial dataset represents "
        "one Baju Melayu collection; it does not contain separate product IDs."
    )


if page == "Product & Chat":
    if st.session_state.selected_product is None:
        st.subheader("Choose a product collection")
        st.write("Select the collection represented in the review dataset to begin your enquiry.")
        c1, c2 = st.columns([1.15, 1])

        with c1:
            st.markdown(
                '<div class="product-card"><div style="font-size:3rem">👔</div>'
                '<h3>Baju Melayu Collection</h3>'
                '<p>Customer experiences about fabric, comfort, sizing, stitching, '
                'colour, delivery and seller service.</p>'
                '<p class="small-muted">Source: Pheiying · W Baju Melayu Shopee reviews</p></div>',
                unsafe_allow_html=True,
            )
            if st.button("Explore this collection →", type="primary", use_container_width=True):
                st.session_state.selected_product = "Baju Melayu Collection"
                st.rerun()

        with c2:
            st.info(
                "Why only one collection? The source CSV has review text and ratings, "
                "but no product ID or product-category field for splitting the reviews "
                "into multiple selectable products."
            )
            st.caption("We avoid presenting categories that the data cannot substantiate.")

    else:
        top1, top2 = st.columns([4, 1])
        with top1:
            st.subheader("👔 Baju Melayu Collection")
            st.caption("Ask about this collection using English, BM or Rojak.")
        with top2:
            if st.button("← Change product"):
                st.session_state.selected_product = None
                st.session_state.messages = []
                st.session_state.voice_transcript = ""
                st.rerun()

        prompts = [
            "Kain dia panas tak?",
            "Saiz dia accurate ke?",
            "Is the material comfortable?",
            "Any complaints about stitching or quality?",
            "Warna sama macam gambar tak?",
            "Delivery usually cepat tak?",
        ]
        st.markdown("**Not sure what to ask? Try one of these:**")
        cols = st.columns(3)
        for i, prompt in enumerate(prompts):
            if cols[i % 3].button(prompt, key=f"prompt_{i}", use_container_width=True):
                st.session_state.pending_question = prompt

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("evidence"):
                    with st.expander("Review evidence used"):
                        for review in msg["evidence"]:
                            st.markdown(
                                f"- **{review['sentiment']} · {review['rating']}★** — "
                                f"{review['review']}"
                            )

        question = st.chat_input("Type your question in English, BM or Rojak...")
        if st.session_state.pending_question:
            question = st.session_state.pending_question
            st.session_state.pending_question = ""

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Finding relevant customer reviews..."):
                    result = assistant.answer(question, top_k=5)

                st.markdown(result["answer"])
                if result["evidence"]:
                    with st.expander("See supporting customer reviews"):
                        for review in result["evidence"]:
                            st.markdown(
                                f"- **{review['sentiment']} · {review['rating']}★** — "
                                f"{review['review']}"
                            )
                st.caption(
                    f"Retrieval: {result['method']} · "
                    f"{len(result['evidence'])} review(s) returned. "
                    "This is customer opinion, not verified product specification."
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "evidence": result["evidence"],
                }
            )
            st.rerun()

        
st.divider()
st.markdown("**🎙️ Voice input (speech-to-text)**")
st.caption(
    "Record your question, check the transcript, "
    "then send it to Rojak."
)

voice_language = st.selectbox(
    "Speech language",
    ["ms-MY", "en-US"],
    format_func=lambda x: (
        "Bahasa Melayu" if x == "ms-MY" else "English"
    ),
    key="voice_language",
)

audio = st.audio_input(
    "Record your question",
    sample_rate=16000,
    key="voice_input",
)

if audio is not None:
    audio_bytes = audio.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    if audio_hash != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = audio_hash

        with st.spinner("Transcribing your question..."):
            try:
                st.session_state.voice_transcript = (
                    transcribe_recording(
                        audio_bytes,
                        voice_language
                    )
                )
            except sr.UnknownValueError:
                st.warning(
                    "Sorry, I couldn't understand that. "
                    "Please try again."
                )
            except sr.RequestError as exc:
                st.error(f"Google speech recognition error: {exc}")
            except Exception as exc:
                st.error(f"Audio error: {exc}")

if st.session_state.voice_transcript:
    st.success(
        f"Recognised: {st.session_state.voice_transcript}"
    )

    if st.button("Send recognised question", type="primary"):
        st.session_state.pending_question = (
            st.session_state.voice_transcript
        )
        st.session_state.voice_transcript = ""
        st.rerun()



elif page == "Review Insights":
    st.subheader("Review insights")
    st.caption(
        "Descriptive analysis of the selected dataset. Ratings are used as sentiment "
        "proxies in this initial build."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviews", len(df))
    c2.metric("Average rating", f"{df['rating'].mean():.2f}/5")
    c3.metric("Unique authors", df["author"].nunique())

    counts = assistant.sentiment_distribution()
    left, right = st.columns(2)
    with left:
        st.markdown("**Rating-derived sentiment distribution**")
        st.bar_chart(counts)
    with right:
        st.markdown("**Rating distribution**")
        st.bar_chart(df["rating"].value_counts().sort_index())

    st.warning(
        "Sentiment labels are derived from star ratings (1–2 negative, 3 neutral, "
        "4–5 positive). They are proxy labels, not human-annotated sentiment labels. "
        "Review text can express mixed sentiment despite its rating."
    )
    st.markdown("**Recent / sample reviews**")
    st.dataframe(
        df[["rating", "review"]].head(30),
        use_container_width=True,
        hide_index=True,
    )


elif page == "NLP Lab & Syllabus":
    st.subheader("NLP Lab")
    st.write(
        "These tools demonstrate syllabus techniques on the review corpus. "
        "Some are exploratory rather than part of the live answer pipeline."
    )
    tab1, tab2, tab3 = st.tabs(
        ["Preprocessing & morphology", "POS & WSD", "Topic modelling & Word2Vec / Markov"]
    )

    with tab1:
        sample = st.text_area(
            "Enter an English / BM / Rojak sample",
            "Kain dia cantik gila but size agak besar 😍",
        )
        if st.button("Preprocess sample"):
            info = assistant.preprocess_demo(sample)
            st.write("**REGEX-cleaned:**", info["cleaned"])
            st.write("**Tokens:**", info["tokens"])
            st.write(
                "**Stemmed (Malay-capable if Sastrawi installed):**",
                info["stems"],
            )

    with tab2:
        sentence = st.text_input(
            "Sentence for POS tagging",
            "The kain is soft and comfortable",
        )
        if st.button("Run POS tagging"):
            st.write(assistant.pos_tag(sentence))

        ambiguous = st.text_input("Word for Word Sense Disambiguation (Lesk)", "bank")
        context = st.text_input(
            "Context sentence",
            "I deposited money at the bank",
        )
        if st.button("Try Lesk WSD"):
            st.write(assistant.wsd(ambiguous, context))

        st.caption(
            "POS tagging and Lesk WSD are demonstrations; NLTK resources may need "
            "to be downloaded on first use. Lesk may be weak on code-mixed text."
        )

    with tab3:
        n_topics = st.slider("Number of LDA topics", 2, 6, 3)
        if st.button("Run LDA topic modelling"):
            with st.spinner("Modelling review topics..."):
                st.write(assistant.lda_topics(n_topics))
        if st.button("Train Word2Vec exploration"):
            st.write(assistant.train_word2vec())
        if st.button("Generate Markov text demo"):
            st.write(assistant.markov_demo())

        st.caption(
            "LDA, Word2Vec and Markov NLG are corpus-based syllabus demonstrations. "
            "Markov output is experimental and is not used to answer shopper questions."
        )


elif page == "Feedback & Learning":
    st.subheader("Human correction and learning")
    st.write(
        "If an answer or retrieved review is unsuitable, submit a correction. "
        "Corrections are saved locally and can be included in a future sentiment-model "
        "retraining run."
    )

    with st.form("correction_form"):
        original = st.text_area("Original user enquiry")
        corrected = st.text_area("Corrected / preferred response or note")
        label = st.selectbox(
            "Correct sentiment label (optional)",
            ["Not provided", "positive", "neutral", "negative"],
        )
        submitted = st.form_submit_button("Save correction")

        if submitted:
            save_correction(
                original,
                corrected,
                None if label == "Not provided" else label,
            )
            st.success(
                "Correction saved to corrections.csv. "
                "It is not automatically treated as a verified product fact."
            )

    corr_path = Path("corrections.csv")
    if corr_path.exists():
        st.markdown("**Saved corrections**")
        st.dataframe(
            pd.read_csv(corr_path),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("**Sentiment model evaluation**")
    st.caption(
        "Ratings provide weak labels: 1–2 negative, 3 neutral, 4–5 positive. "
        "Evaluation is a baseline, not a gold-standard human-labelled benchmark."
    )

    if st.button("Train and evaluate sentiment baseline"):
        with st.spinner(
            "Training TF-IDF + Logistic Regression and evaluating held-out reviews..."
        ):
            metrics = assistant.evaluate_sentiment()

        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        st.text(metrics["report"])
        st.write("**Confusion matrix**")
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=metrics["labels"],
                columns=metrics["labels"],
            )
        )
