# Modelo Bert de Perguntas e Respostas

#### Aluno: Nelson Custódio da Silveira Neto (https://github.com/ncsneto)
#### Orientador(/a/es/as): PhD Leonardo Alfredo Forero Mendoza (prof.leonardo@ica.ele.puc-rio.br) e Cristian Munoz (prof.cristian@ica.ele.puc-rio.br).

---

Trabalho apresentado ao curso [BI MASTER](https://ica.puc-rio.ai/bi-master) como pré-requisito para conclusão de curso e obtenção de crédito na disciplina "Projetos de Sistemas Inteligentes de Apoio à Decisão".

- [Link para o código](https://github.com/ncsneto/Question-Answer_BERT). 

- Trabalhos relacionados:
    - [Google Research Project](https://github.com/google-research/bert#pre-trained-models).
    - [Building a QA System with BERT on Wikipedia](https://qa.fastforwardlabs.com/pytorch/hugging%20face/wikipedia/bert/transformers/2020/05/19/Getting_Started_with_QA.html#QA-on-Wikipedia-pages).
    - [Stanford Question Answering Dataset] (https://rajpurkar.github.io/SQuAD-explorer/)
    - [PyTorch Question Answering] (https://github.com/kushalj001/pytorch-question-answering#pytorch-question-answering)
    - [LSTM is dead. Long Live Transformers!] (https://www.youtube.com/watch?v=S27pHKBEp30)
    - [Attention Is All You Need] (https://arxiv.org/abs/1706.03762) (https://www.youtube.com/watch?v=iDulhoQ2pro)
    - [Transformer Neural Networks - EXPLAINED! (Attention is all you need)] (https://www.youtube.com/watch?v=TQQlZhbC5ps&t=13s)
    - [pytorch-question-answering] (https://github.com/kushalj001/pytorch-question-answering/blob/master/1.%20DrQA.ipynb)
    - [BERTimbau - Portuguese BERT] (https://github.com/neuralmind-ai/portuguese-bert)
    - [Portuguese Named Entity Recognition using BERT-CRF] (https://github.com/neuralmind-ai/portuguese-bert)
    - [NLP Tutorial: Creating Question Answering System using BERT + SQuAD on Colab TPU] (https://hackernoon.com/nlp-tutorial-creating-question-answering-system-using-bert-squad-on-colab-tpu-1utp3352)
    - [Portuguese Word Embeddings] (http://www.davidsbatista.net/blog/2019/11/03/Portuguese-Embeddings/)
    - [Pergunta e resposta de BERT] (https://www.tensorflow.org/lite/examples/bert_qa/overview)
    - [Answering Questions With Transformers] (https://predictivehacks.com/answering-questions-with-transformers/)
    - [RAPPORT: A Fact-Based Question Answering System for Portuguese] (https://estudogeral.uc.pt/handle/10316/41880)
    - [Question Answering] (https://paperswithcode.com/task/question-answering)
    - [Fine-tuning a model on a question-answering task] (https://github.com/huggingface/notebooks/blob/master/examples/question_answering.ipynb)
    - [Datasets em Português] (https://forum.ailab.unb.br/t/datasets-em-portugues/251)
    - [Building a QA System with BERT on Wikipedia](https://qa.fastforwardlabs.com/pytorch/hugging%20face/wikipedia/bert/transformers/2020/05/19/Getting_Started_with_QA.html#QA-on-Wikipedia-pages).

---

### Resumo

POC Perguntas e Respostas em português utilizando BERT

Um dos maiores desafios no campo de processamento de linguagem natural (NLP) está na automatização de processos que buscam o entendimento da semântica e compreensão de textos. Modelos de rede neurais aplicados a transferência de aprendizado tem conseguido ótimos resultados na tarefa de identificação de contexto. A arquitetura dos Transformers, um modelo de aprendizado profundo, no qual cada elemento de saída é conectado a cada elemento de entrada, além do mecanismo de “atenção” (Attention) que identifica as ponderações entre eles, foram os responsáveis pela grande evolução e são a base dos diversos modelos de classificação de texto, extração de informações, resposta a perguntas e geração de texto.

Os Transformers foram introduzidos pelo Google em 2017 (paper: Attention Is All You Need). Na época, os modelos de linguagem usavam principalmente redes neurais recorrentes (RNN) e redes neurais convolucionais (CNN) para lidar com tarefas de NLP. Embora esses modelos tenham bons resultados, o Transformer/Attention é considerado uma evolução importante.

Enquanto os modelos convencionais de redes neurais necessitam que as sequências de dados sejam processados em ordem fixa, o modelo Transformer processa os dados em qualquer ordem, permitindo uma eficiência maior no processo de treinamento, além de utilizar estratégias como a MLM (Masked Language Model) que faz com o que o modelo identifique uma palavra “mascarada” baseando-se no contexto em que ela está inserida. Isso permite que o modelo vincule as palavras ao contexto das sentenças em que estão inseridas e com isso responder melhor as pesquisas que são feitas. O mecanismo de atenção (Attention) desempenha um papel importante em enfatizar em qual parte do contexto o modelo deve se concentrar.

Essa flexibilidade permitiu que os modelos Transformes fossem pré-treinados, criando uma camada de "conhecimento", que a partir do processo de transferência de aprendizado, pode ser adaptado aos conteúdos pesquisados (fine tunning) obtendo-se respostas melhores do modelo.

O primeiro modelo lançado (2018) foi o BERT (Bidirectional Encoder Representations from Transformers), com o seu código aberto e uma quantidade considerável de dados pré-treinados. Atualmente existem dezenas de modelos diferentes (RoBERTa, DistilBERT, GTP, GTP-2, etc...) que foram aprimorados pelas grandes empresas do mercado (Google, Facebook, Microsoft) e por startups de tecnologia como Huggingface e OpenAI.

Implementamos nesta prova de conceito de Perguntas e Respostas o modelo BERT, utilizando a biblioteca Transformers da Huggingface (🤗), sobre um texto extraído da wikipidea. Em função do custo computacional para realização do treinamento e o fine tunning de um novo modelo, buscamos modelos já pré-treinados na base da Huggingface (🤗). Os modelos padrões (sem fine tunning) de BERT (large e Base) não tiveram resposta satisfatória. Temos poucas opções de modelos pré-treinados em português, sendo a maioria traduções automatizadas de modelos em inglês, o que limita a utilização de modelos pré-treinados para o Português.

O modelo (mrm8488/bert-base-portuguese-cased-finetuned-squad-v1-pt) disponibilizado pelo usuário Manuel Romero (mrm8488 do github) foi o que trouxe os melhores resultados.

A Hugging Face (🤗) é uma startup focada em NLP com uma grande comunidade de código aberto. Eles desenvolveram uma biblioteca (transformers) baseada em python que disponibiliza uma API para as principais arquiteturas conhecidas, como BERT, RoBERTa, GPT-2 ou DistilBERT, que estão sendo utilizadas, com resultados de última geração em uma variedade de tarefas de NLP como: classificação de texto, extração de informações, resposta a perguntas e geração de texto. Essas arquiteturas já possuem diversos corpus pré-treinados em diversas línguas.


---

Matrícula: 191.671.022

Pontifícia Universidade Católica do Rio de Janeiro

Curso de Pós Graduação *Business Intelligence Master*
