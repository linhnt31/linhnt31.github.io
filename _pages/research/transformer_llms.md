---
title: How Transformer LLMs Work
layout: default
permalink: /research/transformer_llms
published: true
---

# How Transformer LLMs Work

> I have wanted to understand Transformers and related advances in a structured, in-depth way for a long time, but I struggled to find the right resources. I therefore decided to study the topic systematically. These notes synthesize the *How Transformer LLMs Work* and *Attention in Transformers: Concepts and Code in PyTorch* courses by DeepLearning.AI with material from other courses and readings I did. I hope they also help others build a clear understanding of Transformers, and I will continue updating them over time. 

> **NOTE**: At some points, you and (I) will find it difficult, but believe me, you will find the answer by reading further (I hide it from you and future me in later parts @@).  

## Table of Contents

1. [Language Models: Evolution and Fundamentals](#language-model-evolution)
    1. [Bag-of-Words](#bag-of-words)
    2. [Static Word Embeddings](#static-word-embeddings)
    3. [RNNs and Attention](#rnns-and-attention)
2. [Transformer Architecture](#transformer-architecture)
    1. [Transformer Encoder](#transformer-encoder)
    2. [Transformer Decoder](#transformer-decoder)
    3. [Encoder-Only and Decoder-Only Transformers](#encoder-only-and-decoder-only)
    4. [Context Length](#context-length)
3. [Processing Pipeline](#processing-pipeline)
    1. [Tokenization](#tokenization)
    2. [Embeddings](#embeddings)
    3. [Transformer Processing](#transformer-processing)
4. [Self-Attention](#self-attention)
    1. [From Hidden States to Q, K, and V](#attention-head)
    2. [Scaled Dot-Product Attention](#attention-math)
    3. [Multi-Head Attention and Efficient Variants](#attention-variants)
5. [Mixture of Experts](#mixture-of-experts)
    1. [Experts and Routers](#experts-and-routers)
    2. [Total and Active Parameters](#total-and-active-parameters)
    3. [Advantages and Limitations](#moe-advantages-and-limitations)
6. [References](#references)

---

## 1. Language Models: Evolution and Fundamentals {#language-model-evolution}

### 1.1. Bag-of-Words {#bag-of-words}

- We begin with unstructured text.

- Through **tokenization**, we divide the text into tokens, such as words.

- The unique tokens form a **vocabulary**.

- A document can then be represented as a numerical vector containing the count or presence of each vocabulary item.

> **Limitation**: Bag-of-Words does not directly represent word order or semantic meaning.

### 1.2. Static Word Embeddings {#static-word-embeddings}

Neural methods such as **Word2Vec** learn dense vector embeddings that capture semantic relationships between words. We can compare these vectors to estimate the similarity between words.

> **Limitation**: Word2Vec creates **static embeddings**. For example, it produces the same embedding for *bank* in *river bank* and *financial bank*, regardless of the surrounding context.

### 1.3. RNNs and Attention {#rnns-and-attention}

- Recurrent neural networks (RNNs) process a sequence token by token while maintaining a hidden state containing information from previous positions.

- In an encoder-decoder RNN, the encoder represents the input sequence, and the decoder generates the output sequence autoregressively.

> A single fixed-size context vector may fail to preserve all relevant information from a long or complex input sequence.

- Attention allows the decoder to access different encoder states instead of relying on only one context vector. However, the recurrent computation remains sequential, which limits parallelization.

---

## 2. Transformer Architecture {#transformer-architecture}

A Transformer consists of a stack of \(N\) structurally identical layers with separate learned parameters. Each layer uses **attention** to incorporate/attend information from permitted token positions and a **position-wise feed-forward network** to transform each token representation independently. The original encoder–decoder Transformer has separate encoder and decoder stacks, with decoder layers additionally using cross-attention over encoder outputs.

![Transformer encoder-decoder architecture]({{ site.baseurl }}/assets/images/Research/transformer_arc.jpg)

### 2.1. Transformer Encoder {#transformer-encoder}

The Transformer encoder converts input token embeddings, which are incorporated with positional information, into contextualized representations. Each encoder layer contains:

- **Positional embedding** represents the position of each token in a sequence. In models that use additive positional embeddings, each positional vector has the same dimension as the corresponding token embedding. Positional information is necessary because *self-attention does not inherently represent token order*.

- **Multi-head self-attention**: we can have multiple heads, with different weight sets of queries, keys, and values (which will be shown in Section [4](#self-attention) given the same input token embeddings. With multi-head self-attention, Transformer let each token to collect relevant context information from other tokens in the input sequence.

- **Position-wise feed-forward network (PFFN)**: applies the same nonlinear transformation to *each contextualized token representation* independently.

    + 


### 2.2. Transformer Decoder {#transformer-decoder}

In the **original** encoder-decoder Transformer, the decoder generates the output sequence using previously available output tokens and the encoder's contextualized input representations. Each decoder layer contains:

- **Masked self-attention**: applies a causal mask so that each position can attend only to itself and earlier positions.

- **Encoder-decoder attention** or **Cross-attention** (in traditional encoder-decoder transformer): *uses decoder representations* (i.e., output of the masked self-attention, which will be detailed in next sections) as **queries** and encoder outputs as **keys** and **values**. You can look at the sub-figure 4 below, where we have an example for using cross attention for *multimodal*.

> Later, researchers have found out that using encoder-only and decoder-only can work well for different applications.

<img src="{{ site.baseurl }}/assets/images/Research/cross-attention.jpg" alt="Source: https://www.linkedin.com/posts/cwolferesearch_cross-attention-is-a-fundamental-idea-that-activity-7310657138467446784-iyvV/">

- **Position-wise feed-forward network**: applies a nonlinear transformation to each token representation independently.

A final linear layer and softmax convert a decoder hidden state into a probability distribution over the vocabulary.


> To avoid confusion among concepts of attention, we can remember them as follows: **Bidirectional self-attention** allows each token to attend to all unmasked tokens (e.g., BERT), **masked/causal self-attention** blocks each token to attend to future tokens (e.g., GPT), **encoder-decoder attention/cross-attention** lets decoder representations (i.e., *queries*) to attend to encoder outputs (*keys* and *values*) (e.g., multimodal).

### 2.3. Encoder-Only and Decoder-Only Transformers {#encoder-only-and-decoder-only}

- **Encoder-only models**, such as BERT, produce *contextualized representations* for all input tokens. During masked-language-model pre-training, BERT masks selected input tokens and learns to predict their original values. It can then be fine-tuned for downstream tasks.

    - BERT uses bidirectional self-attention, so each ordinary token can attend to tokens on both sides.

    - The final representation of `[CLS]` is commonly used for sequence-level tasks after suitable training. `[SEP]` marks the end of a sequence or separates a pair of sequences.

- **Decoder-only models**, such as GPT, use **causal self-attention** and learn to predict the next token from the preceding tokens.

    - Each position can attend only to itself and earlier positions. This **causal attention mask** differs from BERT's token masking: GPT hides future attention connections, whereas BERT replaces selected input tokens and predicts them using context from both sides.

### 2.4. Context Length {#context-length}

The **context length** is the maximum number of tokens that a model can process in one forward pass. During generation, the context contains the prompt and previously generated tokens.

---

## 3. Processing Pipeline {#processing-pipeline}

### 3.1. Tokenization {#tokenization}

A **tokenizer** divides the input into tokens and maps each token to a discrete **token ID**. Tokenizers may operate at the word, subword, character, or byte level. Subword tokenization is widely used.

```text
Text:      "Transformers are useful"
Tokens:    ["Transform", "ers", "are", "useful"]
Token IDs: [5812, 1047, 389, 7421]
```

Each tokenizer has a fixed **vocabulary size**, which determines how many distinct tokens it can represent.

> A larger vocabulary can represent common words or phrases using fewer tokens. However, it also increases the size of the embedding table and, in a language model, the vocabulary-output layer.

### 3.2. Embeddings {#embeddings}

The model uses each token ID to retrieve a learned vector from its embedding table. It then incorporates positional information to distinguish the order of the tokens.

### 3.3. Transformer Processing {#transformer-processing}

Transformer layers convert the input embeddings into contextualized token representations.

- An encoder-only model can use these representations for tasks such as classification, retrieval, and token labeling.

- A decoder-only language model passes its contextualized representations through an LM head to produce next-token probability distributions. During generation, it uses the distribution at the final position to select the next token.

<img src="{{ site.baseurl }}/assets/images/Research/decoder-only.png" alt="Source: Attention in Transformers: Concepts and Code in PyTorch - deeplearning.ai">

---

## 4. Self-Attention {#self-attention}

Self-attention updates each token representation by combining information from other **permitted** positions in the same sequence. Which positions are permitted depends on the attention mask: **bidirectional models** (e.g., BERT) can usually use both earlier and later tokens, while **causal models** (e.g., GPT) cannot use/attend future tokens.

![Workflow from tokenization to self-attention]({{ site.baseurl }}/assets/images/Research/attention.png)

> The diagram shows additive positional information and unrestricted attention for simplicity. Some architectures instead apply position information to $Q$ and $K$; for example, RoPE rotates them before their dot products are calculated. You can find an example for **single-head** (masked) self-attention class below:

```python
class MaskedSelfAttention(nn.Module):                     
    def __init__(self, d_model=2, # dimension for each input token embedding 
                 row_dim=0, # row and column indexes
                 col_dim=1): 
        
        super().__init__()
        
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        self.row_dim = row_dim
        self.col_dim = col_dim

    # This method is to calculate the (masked) self-attention values for each token    
    def forward(self, token_encodings, mask=None):

        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            ## Here we are masking out things we don't want to pay attention to
            ##
            ## We replace values we wanted masked out
            ## with a very small negative number so that the SoftMax() function
            ## will give all masked elements an output value (or "probability") of 0.
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9) # I've also seen -1e20 and -9e15 used in masking

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
```

### 4.1. From Hidden States to $Q$, $K$, and $V$ {#attention-head}

At an attention layer, $X \in \mathbb{R}^{N \times d_{\text{model}}}$ contains one hidden-state vector per token. Particularly, $N$ is the sequence length and $d_{\text{model}}$ is the dimension of each token embedding and hidden representation. In standard multi-head attention with $h$ heads, this dimension is divided among the heads, so each head usually has $d_k = d_{\text{model}}/h$ query and key dimensions .For the first layer, $X$ comes from *token embeddings, with positional information* incorporated according to the architecture. In later layers, it is the output of the preceding layer.

For ***each token*** in one attention head, three learned linear projections produce[^3]:

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V$$

where $W^Q,W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$. Therefore, $Q,K \in \mathbb{R}^{N \times d_k}$ and $V \in \mathbb{R}^{N \times d_v}$.

|  | The question it answers | Its role |
|---|---|---|
| **Query $Q$** | “What am I looking for?” | The token doing the looking |
| **Key $K$** | “What am I?” | How a token advertises itself |
| **Value $V$** | “What do I contribute if chosen?” | The content actually retrieved |

Separate projections let the model learn different representations for matching and for transferring information. In particular, $QK^T$ can express directional relationships, whereas $XX^T$ is symmetric.

> **Training note:** When training from scratch, the token embeddings and projection matrices normally begin with random values and are updated by backpropagation. Learned positional embeddings are updated too; fixed positional encodings are not. By contrast, $X$, $Q$, $K$, and $V$ are temporary **activations**. They are recomputed on every forward pass and change as the model parameters change.

### 4.2. Scaled Dot-Product Attention {#attention-math}

One attention head computes [^1]:

$$S = \frac{QK^T}{\sqrt{d_k}} + M, \qquad A = \operatorname{softmax}(S), \qquad Z = AV$$

- **Score:** $QK^T$ compares every query with every key, answering the question: **How relevant each token to the current token?**. Dividing by $\sqrt{d_k}$ keeps large dot products from saturating the softmax function.

    + Why divide by $\sqrt{d_k}$ [^3]? **Answer**: Each raw attention logit is a dot product containing $d_k$ terms. Under the simplifying assumptions that the query and key components are independent, zero-centered, and have unit variance, the logit has variance $d_k$ and standard deviation $\sqrt{d_k}$. Without scaling, wider heads therefore produce larger gaps between logits, which can ***make the softmax overly concentrated and its gradients very small***. Dividing by $\sqrt{d_k}$ approximately normalizes the logit variance to $1$, ***keeping the softmax scale and gradients more consistent across head dimensions***. This does not make different heads or head counts equivalent; it only ***removes the unintended growth in score magnitude***, while each head can still learn distinct attention patterns.

- **Mask:** $M$ is $0$ for allowed connections and a very large negative value for blocked ones, thereby after softmax function(), the token will have $0\%$ similarity to the token that came after it. It can be omitted when no positions are blocked - *self-attention*. For example, 

<img src="{{ site.baseurl }}/assets/images/Research/attention-mask.jpg" alt="Attention matrix mask (Source: https://krypticmouse.hashnode.dev/attention-is-all-you-need)">

- **Normalize:** Row-wise softmax produces the **attention-weight** matrix $A$. Each row is nonnegative and sums to $1$. *It determines the percentages of influence each token has on other tokens*.

- **Combine:** $AV$ produces $Z$, a weighted sum of value vectors for each query position/token. For each token, relevant tokens get large weights, giving each token its answer [^3].

For self-attention over $N$ tokens, $S,A \in \mathbb{R}^{N \times N}$ and $Z \in \mathbb{R}^{N \times d_v}$.

> A dot product depends on both vector direction and magnitude. Unlike cosine similarity, it is not normalized to the range $[-1,1]$.

### 4.3. Multi-Head Attention and Efficient Variants {#attention-variants}

Transformers normally run several attention heads in parallel. Each head can learn different relationships; their outputs are concatenated and mixed through an output projection $W^O$:

$$\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

- **Multi-head attention (MHA):** gives each head its own query, key, and value projections.
- **Multi-query attention (MQA):** lets all query heads share one key head and one value head, reducing the key-value cache and memory bandwidth during generation.
- **Grouped-query attention (GQA):** lets groups of query heads share key and value heads, providing a compromise between MHA and MQA.
- **Sparse attention:** restricts each query to selected positions, reducing the cost of long sequences.
- **Ring Attention:** distributes long-sequence attention across devices by circulating blocks of keys and values.

---

## 5. Mixture of Experts {#mixture-of-experts}

**Mixture of Experts (MoE)** [^2] increases a model's parameter capacity without activating all parameters for every token. In selected Transformer layers, MoE replaces the traditional FFNN with multiple FFNNs, called **experts**.

![MoE overview]({{ site.baseurl }}/assets/images/Research/moe.png)

### 5.1. Experts and Routers {#experts-and-routers}

- **Intuition**: each MoE layer has a set of experts. A **router** selects a small subset of them to process each token representation. Different experts may learn different patterns, although their specializations are not always easy to interpret.

- **Expert**: is normally an FFNN within an MoE layer.

- **Router**: is a small learned gating function, often a linear layer, that assigns each expert a score for a given token representation. It selects, for example, the top-$k$ highest-scoring experts and combines their outputs.

![MoE layer with a router and experts]({{ site.baseurl }}/assets/images/Research/router_moe.png)

### 5.2. Total and Active Parameters {#total-and-active-parameters}

- **Total parameters** include the input embeddings, attention layers, routers, all experts, and the LM head. During standard inference, all experts must be available in memory, although they may be distributed across multiple devices.

- **Active parameters** are the parameters used to process a particular token. Only the selected top-$k$ experts are activated for that token. This is why MoE is called a **sparsely activated model**.

- MoE requires less computation than a dense model with a similar total number of parameters. However, it may still require more computation than a smaller dense model.

### 5.3. Advantages and Limitations {#moe-advantages-and-limitations}

**Advantages**:

1. Increases model capacity without increasing computation in direct proportion to the total parameter count.
2. Allows different experts to learn different patterns.
3. Can improve performance over dense models under a similar computational budget.

**Limitations**:

1. Requires substantial memory to store all experts.
2. Introduces routing and communication overhead.
3. May overload some experts while leaving others under-trained.
4. Is more complex to train and deploy than a dense model.

---

## 6. References {#references}

[^1]: [The Math Behind Multi-Head Attention in Transformers](https://medium.com/data-science/the-math-behind-multi-head-attention-in-transformers-c26cba15f625)

[^2]: [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)

[^3]: [LLM Architecture Refresh: Inside a Transformer Block — Attention, Heads, and the FFN](https://bearbearyu1223.github.io/posts/llm-architectures-attention-and-rope/#taking-a-transformer-block-apart-one-measurement-at-a-time)
