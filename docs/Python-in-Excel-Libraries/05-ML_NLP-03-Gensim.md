# Leveraging Gensim in Python in Excel

Gensim is a robust Python library for topic modeling, document similarity, and natural language processing (NLP). With Python in Excel, you can use Gensim to perform advanced text analytics—such as building word embeddings, topic models, and similarity queries—directly within your spreadsheets.

## 1. Setup and Imports

To use Gensim, reserve the first worksheet for import statements:

```python
=PY(
import gensim
from gensim import corpora, models, similarities
)
```

This makes the core Gensim modules available for all subsequent Python cells.

## 2. Preparing Text Data

- **Tokenize and clean text:**

```python
=PY(
documents = xl("TextData[#All]", headers=True)
tokenized = [doc.lower().split() for doc in documents]
tokenized
)
```

- **Create a dictionary and corpus:**

```python
=PY(
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(text) for text in tokenized]
)
```

## 3. Topic Modeling with LDA

```python
=PY(
lda = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)
topics = lda.print_topics(num_words=5)
topics
)
```

## 4. Document Similarity

- **Build similarity index:**

```python
=PY(
index = similarities.MatrixSimilarity(lda[corpus])
sims = index[lda[corpus[0]]]
sims
)
```

## 5. Word Embeddings

- **Train Word2Vec model:**

```python
=PY(
from gensim.models import Word2Vec
model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1, workers=2)
model.wv.most_similar('python')
)
```

## 6. Best Practices

- **Imports Once**: Place all import statements on the first worksheet.
- **Data Preparation**: Clean and tokenize text before modeling.
- **Output Management**: Return lists or DataFrames for easy review in Excel.
- **Performance**: For large corpora, sample or preprocess data to maintain responsiveness.

By leveraging Gensim in Python in Excel, you can perform advanced NLP and topic modeling tasks directly in your spreadsheets, making sophisticated text analytics accessible to all Excel users.
