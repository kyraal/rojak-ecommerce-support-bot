import pandas as pd
from nlp_engine import ReviewAssistant, clean_text

def test_clean_text():
    assert 'panas' in clean_text('KAIN PANAS!!! 😍')

def test_retrieval():
    df = pd.DataFrame({
        'review': ['Kain lembut dan selesa', 'Saiz baju agak besar', 'Penghantaran lambat'],
        'rating': [5, 3, 1],
        'author': ['a', 'b', 'c'],
        'sentiment': ['positive', 'neutral', 'negative']
    })
    bot = ReviewAssistant(df)
    assert bot.retrieve('kain selesa', top_k=2)
