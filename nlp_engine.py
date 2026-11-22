import re, html, json, random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _find_col(cols, candidates):
    lower={str(c).strip().lower():c for c in cols}
    for c in candidates:
        if c in lower: return lower[c]
    for key,val in lower.items():
        if any(c in key for c in candidates): return val
    return None


def load_reviews(path):
    df=pd.read_csv(path,encoding='utf-8-sig')
    text_col=_find_col(df.columns,['review','reviews','text','comment','content'])
    rating_col=_find_col(df.columns,['rating','ratings','stars','score'])
    author_col=_find_col(df.columns,['author','user','username','user id','userid'])
    if text_col is None: raise ValueError(f"Could not identify review-text column. Columns found: {list(df.columns)}")
    out=pd.DataFrame({'review':df[text_col].fillna('').astype(str)})
    out['rating']=pd.to_numeric(df[rating_col],errors='coerce') if rating_col else np.nan
    out['author']=df[author_col].fillna('Unknown').astype(str) if author_col else 'Unknown'
    out=out[out.review.str.strip().ne('')].copy().reset_index(drop=True)
    out['rating']=out['rating'].clip(1,5)
    out['sentiment']=out.rating.apply(rating_to_sentiment)
    return out


def rating_to_sentiment(r):
    if pd.isna(r): return 'unknown'
    if r<=2: return 'negative'
    if r==3: return 'neutral'
    return 'positive'


def clean_text(text):
    text=html.unescape(str(text)).lower()
    text=re.sub(r'https?://\S+|www\.\S+',' ',text)
    text=re.sub(r'@[\w.]+',' ',text)
    text=re.sub(r'[^\w\s!?.,]',' ',text,flags=re.UNICODE)
    text=re.sub(r'(.)\1{2,}',r'\1\1',text)
    text=re.sub(r'\s+',' ',text).strip()
    return text


class ReviewAssistant:
    def __init__(self,df):
        self.df=df.copy()
        self.df['clean']=self.df.review.map(clean_text)
        self.word=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,strip_accents='unicode')
        self.char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,sublinear_tf=True,max_features=30000)
        self.word_matrix=self.word.fit_transform(self.df.clean)
        self.char_matrix=self.char.fit_transform(self.df.clean)

    def retrieve(self,question,top_k=5):
        q=clean_text(question)
        if not q: return []
        qword=self.word.transform([q]); qchar=self.char.transform([q])
        scores=.55*cosine_similarity(qword,self.word_matrix).ravel()+.45*cosine_similarity(qchar,self.char_matrix).ravel()
        order=np.argsort(scores)[::-1]
        results=[]
        for idx in order[:top_k]:
            if scores[idx] < .025: continue
            r=self.df.iloc[int(idx)]
            results.append({'review':r.review.strip(),'rating':None if pd.isna(r.rating) else int(r.rating),'sentiment':r.sentiment,'score':float(scores[idx])})
        return results

    def answer(self,question,top_k=5):
        hits=self.retrieve(question,top_k)
        if not hits:
            return {'answer':"I couldn't find sufficiently relevant information in the available customer reviews. Try asking about material, comfort, sizing, stitching, colour, delivery or seller experience.",'evidence':[],'method':'word + character TF-IDF cosine similarity'}
        texts=[h['review'] for h in hits]
        sentiments=Counter(h['sentiment'] for h in hits if h['sentiment']!='unknown')
        total=sum(sentiments.values())
        if total:
            dominant=sentiments.most_common(1)[0][0]
            pos=sentiments.get('positive',0); neg=sentiments.get('negative',0); neu=sentiments.get('neutral',0)
            if pos and neg: tone='mixed'
            elif dominant=='positive': tone='mostly positive'
            elif dominant=='negative': tone='mostly negative'
            else: tone='mixed or neutral'
        else: tone='not enough rating information'
        summary=self._theme_summary(texts)
        answer=(f"**Based on the most relevant customer reviews, feedback is {tone}.**\n\n"
                f"{summary}\n\n"
                f"I found {len(hits)} relevant review(s). These are individual customer experiences, so results may vary. Check the supporting reviews below before making a decision.")
        return {'answer':answer,'evidence':hits,'method':'word + character TF-IDF cosine similarity'}

    def _theme_summary(self,texts):
        # English-language summary frame built from retrieved evidence; no external generative model.
        combined=' '.join(clean_text(t) for t in texts)
        terms=[('material / fabric',['kain','material','fabric','cotton','crepe']),('comfort / heat',['selesa','comfort','comfortable','panas','hot','sejuk','cool']),('sizing / fit',['saiz','size','fit','muat','ukuran','measurement']),('stitching / quality',['jahitan','stitch','quality','koyak','kualiti','hole']),('colour / appearance',['warna','colour','color','cantik','design']),('delivery',['delivery','penghantaran','shipping','sampai','pos']),('seller service',['seller','servis','service','response','respond'])]
        found=[label for label,words in terms if any(w in combined for w in words)]
        if found:
            return "The retrieved reviews discuss " + ', '.join(found[:-1]) + (" and " if len(found)>1 else "") + (found[-1] if found else "") + ". The review excerpts provide the specific positive and negative experiences behind this summary."
        return "The retrieved reviews contain related customer comments, but the available text does not support a more specific theme summary."

    def sentiment_distribution(self):
        return self.df.sentiment.value_counts().reindex(['positive','neutral','negative','unknown'],fill_value=0)

    def preprocess_demo(self,text):
        cleaned=clean_text(text); tokens=re.findall(r"[a-zA-Z]+|[0-9]+",cleaned)
        try:
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
            stemmer=StemmerFactory().create_stemmer(); stems=[stemmer.stem(t) for t in tokens]
        except Exception: stems=tokens
        return {'cleaned':cleaned,'tokens':tokens,'stems':stems}

    def pos_tag(self,text):
        try:
            import nltk
            try: nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            except LookupError: nltk.download('averaged_perceptron_tagger_eng',quiet=True)
            try: nltk.data.find('tokenizers/punkt_tab')
            except LookupError: nltk.download('punkt_tab',quiet=True)
            return nltk.pos_tag(nltk.word_tokenize(text))
        except Exception as e: return f"POS tagging unavailable: {e}"

    def wsd(self,word,context):
        try:
            import nltk
            try: nltk.data.find('corpora/wordnet')
            except LookupError: nltk.download('wordnet',quiet=True)
            from nltk.wsd import lesk
            syn=lesk(context.split(),word)
            return {'synset':str(syn),'definition':syn.definition() if syn else 'No sense found'}
        except Exception as e: return f"WSD unavailable: {e}"

    def lda_topics(self,n_topics=3):
        vec=CountVectorizer(stop_words='english',ngram_range=(1,2),max_df=.95,min_df=2,max_features=1500)
        X=vec.fit_transform(self.df.clean)
        if X.shape[1]==0: return 'Not enough vocabulary for topic modelling.'
        from sklearn.decomposition import LatentDirichletAllocation
        lda=LatentDirichletAllocation(n_components=n_topics,random_state=42,learning_method='batch',max_iter=8)
        lda.fit(X); names=vec.get_feature_names_out(); result=[]
        for i,topic in enumerate(lda.components_): result.append({'topic':i+1,'keywords':[names[j] for j in topic.argsort()[-8:][::-1]]})
        return result

    def train_word2vec(self):
        try:
            from gensim.models import Word2Vec
            sentences=[re.findall(r'\w+',t.lower()) for t in self.df.clean if t.strip()]
            model=Word2Vec(sentences,vector_size=50,window=4,min_count=2,workers=1,epochs=30)
            words=list(model.wv.index_to_key[:20])
            return {'vocabulary_size':len(model.wv),'sample_words':words,'note':'Exploratory embeddings trained on this small review corpus; not used for live retrieval.'}
        except Exception as e: return f"Word2Vec unavailable: {e}"

    def markov_demo(self):
        words=' '.join(self.df.review.astype(str)).split()
        if len(words)<3: return 'Not enough text.'
        chain={}
        for a,b,c in zip(words,words[1:],words[2:]): chain.setdefault((a.lower(),b.lower()),[]).append(c)
        starts=list(chain)
        if not starts:return 'Not enough text.'
        pair=random.choice(starts); out=[pair[0],pair[1]]
        for _ in range(18):
            nxt=chain.get((out[-2].lower(),out[-1].lower()))
            if not nxt:break
            out.append(random.choice(nxt))
        return ' '.join(out)+"\n\n(Experimental Markov output; not a factual chatbot answer.)"

    def evaluate_sentiment(self):
        data=self.df[self.df.sentiment.isin(['positive','neutral','negative'])].copy()
        counts=data.sentiment.value_counts()
        if len(data)<15 or counts.min()<2: return {'accuracy':0.0,'report':'Not enough labelled examples in each class for a meaningful stratified split.','confusion_matrix':[[0]],'labels':['insufficient data']}
        X_train,X_test,y_train,y_test=train_test_split(data.review,data.sentiment,test_size=.25,random_state=42,stratify=data.sentiment)
        model=Pipeline([('features',FeatureUnion([('word',TfidfVectorizer(ngram_range=(1,2),min_df=1)),('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,max_features=25000))])),('clf',LogisticRegression(max_iter=1000,class_weight='balanced'))])
        # Human-labelled corrections may supplement training data only; the held-out test split stays untouched.
        corrections_path=Path('corrections.csv')
        if corrections_path.exists():
            try:
                corrections=pd.read_csv(corrections_path).dropna(subset=['sentiment_label'])
                corrections=corrections[corrections.sentiment_label.isin(['positive','neutral','negative'])]
                if not corrections.empty:
                    X_train=pd.concat([X_train, corrections['question'].fillna('').astype(str)], ignore_index=True)
                    y_train=pd.concat([y_train, corrections['sentiment_label'].astype(str)], ignore_index=True)
            except Exception:
                pass
        model.fit(X_train,y_train); pred=model.predict(X_test); labels=['negative','neutral','positive']
        return {'accuracy':accuracy_score(y_test,pred),'report':classification_report(y_test,pred,labels=labels,zero_division=0),'confusion_matrix':confusion_matrix(y_test,pred,labels=labels).tolist(),'labels':labels}


def save_correction(question,response,label):
    path=Path('corrections.csv'); row=pd.DataFrame([{'question':question,'correction':response,'sentiment_label':label or ''}])
    row.to_csv(path,mode='a',header=not path.exists(),index=False,encoding='utf-8-sig')
