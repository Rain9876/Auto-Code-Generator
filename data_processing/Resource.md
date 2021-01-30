## Notes:

The tycoon for NL-Code research is Microsoft

CodeXGLUE is the BenchMark
https://microsoft.github.io/CodeXGLUE/
https://github.com/microsoft/CodeXGLUE



## Tasks

**Code search** (CodeSearchNet, AdvTest; StacQC, WebQueryTest). A model is given the task of measuring the semantic similarity between text and code. In the retrieval scenario, a test set is newly created where function names and variables in test sets are replaced to test the generalization ability of a model. In text-code classification scenario, a test set where natural language queries come from Bing query log is created to test on real user queries.

**Text-to-code generation** (CONCODE). A model is given the task to generate a code given natural language description. An existing dataset is included.

**Code completion** (PY150, GitHub Java Corpus). A model is tasked with predicting following tokens given a code context. Both token-level and line-level completion are covered. The token-level task is analogous to language modeling, and we include two influential datasets here. Line-level datasets are newly created to test a model’s ability to autocomplete a line.

**Documentation translation** (Microsoft Docs). A model is given the task to translate code documentation between human languages. A dataset, focusing on low-resource multilingual translation, is newly created.





## Dataset
**CodeSearchNet**
(The primary dataset consists of 2 million (comment, code) pairs from open source libraries. Concretely, a comment is a top-level function or method comment (e.g. docstrings in Python), and code is an entire function or method. Currently, the dataset contains Python, Javascript, Ruby, Go, Java, and PHP code. Throughout this repo, we refer to the terms docstring and query interchangeably.)

https://wandb.ai/github/CodeSearchNet/benchmark

https://github.com/github/CodeSearchNet/blob/master/notebooks/ExploreData.ipynb



**AdvTest** :
	Not Found

**StaQC**: A Systematically Mined Question-Code Dataset from Stack Overflow
https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset

**WebQueryTest**: Search4Code (for C# and Java)
	May not correct

**Concode** (may for java)
https://github.com/sriniiyer/concode



**CoNaLa**
https://conala-corpus.github.io/

**BigQuery Stack Overflow Data** 
https://www.kaggle.com/stackoverflow/stackoverflow

**BigQuery Github Data** Already processed







## Models

Baseline 
we provide three baseline models to support these tasks, including a BERT-style pretrained model (in this case, CodeBERT), which is good at understanding problems. We also include a GPT-style pretrained model, which we call CodeGPT, to support completion and generation problems. Finally, we include an Encoder-Decoder framework that supports sequence-to-sequence generation problems



CodeBERT: https://arxiv.org/pdf/2002.08155.pdf


CodeGPT


IntelliCode: https://arxiv.org/pdf/2005.08025.pdf





## Pkg

AST (Abstract Syntax Tree)

fastBPE (generate and apply BPE codes)

Moses (scripts to clean and tokenize text only - no installation required)

Apex (for fp16 training)

Pycfg: (for control flow)



## Literatures
https://www.microsoft.com/en-us/research/blog/codexglue-a-benchmark-dataset-and-open-challenge-for-code-intelligence/





## Yurun's Opinions:

Problems in PyMT5:

1. Use python methods as pre-training data. 

   This may be helpful for limited DocString generation, as Input is Python Code. However, for us code generation, DocString is too little and Code generation is much more than DocString. It shows the situation that output is much more than that of Input. (This is not quite normal, the length of output is less or equal than that of input as usual. Should we extend Max_len of output?) The model is much hard to generate. Therefore, using python methods as pre-training data is not enough. We need pretraining NL as well.

2. Logic behinds generated code is not considered.

   The Logic of generated code is not examinated. The grammer of code is much more strict than Natural Languages. The control flow should be considered.

Solutions:

1. Instead of altering the transformer model, in fact, changing the pretraining method is more realistic.

   For pretraining:

   Input: entire python file instead of python methods in pymt5

   Output: predicted sentences (lines), considering the number of tokens of a line is very likely much less than of sentences. (Need verify later)

   Objectives: (Multi or single Objectives ?, replace token, loss of control flow [More weights for ])

   Approach: 

   1. Take several method extracted sentences (Lines) and given the front and backwards codes (GSP)

   2. Take all logical stmt and let the model to predict (like MLM, but included in the step1), more weights given

      Loss_back = Loss(1) + Loss(1)

2. Control flow 

   For finetuning:

   Actor-Critic Algorithm Or RL with Logic loss 



