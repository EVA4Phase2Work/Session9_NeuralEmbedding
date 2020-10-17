# Session9_NeuralEmbedding


For this Project a  machine learning model is built to detect sentiment (i.e. detect if a sentence is positive or negative) using PyTorch and TorchText. This will be done on movie reviews, using the IMDb dataset.

In this first notebook, we'll start very simple to understand the general concepts whilst not really caring about good results. Further notebooks will build on this knowledge and we'll actually get good results.

Introduction
We'll be using a recurrent neural network (RNN) as they are commonly used in analysing sequences. An RNN takes in sequence of words, $X=\{x_1, ..., x_T\}$, one at a time, and produces a hidden state, $h$, for each word. We use the RNN recurrently by feeding in the current word $x_t$ as well as the hidden state from the previous word, $h_{t-1}$, to produce the next hidden state, $h_t$.

$$h_t = \text{RNN}(x_t, h_{t-1})$$
Once we have our final hidden state, $h_T$, (from feeding in the last word in the sequence, $x_T$) we feed it through a linear layer, $f$, (also known as a fully connected layer), to receive our predicted sentiment, $\hat{y} = f(h_T)$.

Below shows an example sentence, with the RNN predicting zero, which indicates a negative sentiment. The RNN is shown in orange and the linear layer shown in silver. Note that we use the same RNN for every word, i.e. it has the same parameters. The initial hidden state, $h_0$, is a tensor initialized to all zeros.



Note: some layers and steps have been omitted from the diagram, but these will be explained later.

Preparing Data
One of the main concepts of TorchText is the Field. These define how your data should be processed. In our sentiment classification task the data consists of both the raw string of the review and the sentiment, either "pos" or "neg".

The parameters of a Field specify how the data should be processed.

We use the TEXT field to define how the review should be processed, and the LABEL field to process the sentiment.

Our TEXT field has tokenize='spacy' as an argument. This defines that the "tokenization" (the act of splitting the string into discrete "tokens") should be done using the spaCy tokenizer. If no tokenize argument is passed, the default is simply splitting the string on spaces.

LABEL is defined by a LabelField, a special subset of the Field class specifically used for handling labels. We will explain the dtype argument later.

For more on Fields, go here.

We also set the random seeds for reproducibility.
