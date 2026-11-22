# Rojak — Malaysian Rojak Product Enquiry Assistant (Initial Build)

A Streamlit prototype for the NLP assignment. Users select the Baju Melayu collection represented by Pheiying's Shopee review dataset, ask questions in English/BM/Rojak, retrieve relevant reviews using word- and character-level TF-IDF + cosine similarity, and receive a concise English evidence-based summary.

## Dataset setup

Download `dataset.csv` from:

https://github.com/pheiying/Sentiment-Analysis-on-Out-Of-Vocabulary-OOV-Malaysia-Rojak-Language/blob/main/Dataset/dataset.csv

Save it as `data/dataset.csv` in this project. The app detects common column names; the Pheiying file uses `No.`, `Author`, `Rating`, `Review`.

**Dataset scope:** this source represents reviews for W Baju Melayu on Shopee. The CSV does not include a product ID/category field, so the first build intentionally offers one collection rather than inventing multiple product listings.

## Run locally (macOS / Windows)

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\activate    # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Main user flow

1. Select **Baju Melayu Collection**.
2. Choose a suggested prompt or type a question in English, BM or Rojak.
3. Optional: record a voice question; Google speech recognition is online and uses `ms-MY` as a best-effort setting. Mixed-language transcription may be imperfect.
4. Preprocessing cleans text and retains useful informal spellings.
5. Reviews are ranked using a blend of word n-gram and character n-gram TF-IDF cosine similarity.
6. The assistant summarises the retrieved themes in English and shows the original review evidence.
7. Users can submit corrections in Feedback & Learning.

## NLP / assignment coverage

| Requirement / syllabus | Initial implementation | Notes |
|---|---|---|
| Speech-to-text | Optional microphone + SpeechRecognition Google recognizer | Requires microphone permission and internet; code-switching can be imperfect. |
| Slang / informal language | Character n-gram TF-IDF retrieval + regex normalisation | No manually hardcoded slang dictionary. This handles spelling variation statistically, not guaranteed slang understanding. |
| Human correction | Feedback form writes `corrections.csv` | Saved feedback is not automatically a verified fact or automatically retrained in this build. |
| Knowledge-based answers | Review corpus retrieval + cosine similarity + evidence display | Answers are bounded by the customer reviews, not seller policy or live order data. |
| REGEX / preprocessing | URL, mention, punctuation, repeated-character and whitespace handling | Preserves word content and emoji-independent text. |
| Morphology | Optional Sastrawi stemming demo | Malay stemming only; not applied to retrieval by default to preserve original forms. |
| POS tagging | NLTK POS demo | Primarily English; code-mixed accuracy limited. |
| Semantic analysis / WSD | NLTK Lesk demo | Demonstration only; weak on Rojak and short contexts. |
| TF-IDF | Word unigrams/bigrams + character n-grams | Core retrieval method. |
| Sentiment analysis | Logistic Regression with word/char TF-IDF | Additional feature. Labels are derived from star ratings (1–2 negative, 3 neutral, 4–5 positive), so they are weak/proxy labels. |
| Evaluation | Stratified hold-out split, accuracy, precision, recall, F1, confusion matrix | Run from Feedback & Learning. Small dataset means results may vary by split. |
| LDA | Topic modelling tab | Exploratory review themes. |
| Word2Vec | Train-on-click demo | Exploratory embeddings; small corpus. |
| Markov NLG | Corpus-based demo | Demonstration only; never used as factual answer generation. |

## Important methodological limitations

- The dataset is a single product-collection review corpus, not a complete e-commerce FAQ database.
- The app cannot verify specifications, stock, prices, return policy, refunds or order tracking. It should not invent these answers.
- The English summary in this prototype is an evidence-grounded template based on retrieved review themes, not a large language model. Original reviews are shown so users can inspect the source.
- Sentiment labels are inferred from star ratings and may disagree with the actual sentiment expressed in a review.
- The correction form stores user feedback locally. A future version should add a controlled correction-review and retraining workflow, keeping a fixed untouched test set to prevent leakage.
- POS, Lesk WSD, LDA, Word2Vec and Markov are supporting syllabus demonstrations, not all part of the live answering path.

## Files

- `app.py` — Streamlit interface and navigation
- `nlp_engine.py` — preprocessing, retrieval, sentiment evaluation and syllabus demonstrations
- `data/dataset.csv` — place the original dataset here
- `corrections.csv` — created locally when users submit feedback (gitignored)

## Testing suggestions

Test questions such as:
- `Kain dia panas tak?`
- `Is the fabric comfortable for kids?`
- `Saiz dia accurate ke?`
- `Any complaints about jahitan?`
- `Warna sama macam gambar tak?`
- `Delivery cepat tak?`

Also test out-of-scope questions such as `Can I get a refund?` or `Where is my parcel?`. The assistant should return insufficient evidence rather than inventing seller policies or order information.
