## POC Perguntas e Respostas em português utilizando BERT

Um dos maiores desafios no campo de processamento de linguagem natural (NLP) está na automatização de processos que buscam o entendimento da semântica e compreensão de textos. Modelos de rede neurais aplicados a transferência de aprendizado tem conseguido ótimos resultados na tarefa de identificação de contexto. A arquitetura dos Transformers, um modelo de aprendizado profundo, no qual cada elemento de saída é conectado a cada elemento de entrada, além do mecanismo de “atenção” (Attention) que identifica as ponderações entre eles, foram os responsáveis pela grande evolução e são a base dos diversos modelos de classificação de texto, extração de informações, resposta a perguntas e geração de texto.

Os Transformers foram introduzidos pelo Google em 2017 (paper: Attention Is All You Need). Na época, os modelos de linguagem usavam principalmente redes neurais recorrentes (RNN) e redes neurais convolucionais (CNN) para lidar com tarefas de NLP. Embora esses modelos tenham bons resultados, o Transformer/Attention é considerado uma evolução importante.

Enquanto os modelos convencionais de redes neurais necessitam que as sequências de dados sejam processados em ordem fixa, o modelo Transformer processa os dados em qualquer ordem, permitindo uma eficiência maior no processo de treinamento, além de utilizar estratégias como a MLM (Masked Language Model) que faz com o que o modelo identifique uma palavra “mascarada” baseando-se no contexto em que ela está inserida. Isso permite que o modelo vincule as palavras ao contexto das sentenças em que estão inseridas e com isso responder melhor as pesquisas que são feitas. O mecanismo de atenção (Attention) desempenha um papel importante em enfatizar em qual parte do contexto o modelo deve se concentrar.

Essa flexibilidade permitiu que os modelos Transformes fossem pré-treinados, criando uma camada de "conhecimento", que a partir do processo de transferência de aprendizado, pode ser adaptado aos conteúdos pesquisados (fine tunning) obtendo-se respostas melhores do modelo.

O primeiro modelo lançado (2018) foi o BERT (Bidirectional Encoder Representations from Transformers), com o seu código aberto e uma quantidade considerável de dados pré-treinados. Atualmente existem dezenas de modelos diferentes (RoBERTa, DistilBERT, GTP, GTP-2, etc...) que foram aprimorados pelas grandes empresas do mercado (Google, Facebook, Microsoft) e por startups de tecnologia como Huggingface e OpenAI.

Implementamos nesta prova de conceito de Perguntas e Respostas o modelo BERT, utilizando a biblioteca Transformers da Huggingface (?), sobre um texto extraído da wikipidea. Em função do custo computacional para realização do treinamento e o fine tunning de um novo modelo, buscamos modelos já pré-treinados na base da Huggingface. Os modelos padrões (sem fine tunning) de BERT (large e Base) não tiveram resposta satisfatória. Temos poucas opções de modelos pré-treinados em português, sendo a maioria traduções automatizadas de modelos em inglês, o que limita a utilização de modelos pré-treinados para o Português.

O modelo (mrm8488/bert-base-portuguese-cased-finetuned-squad-v1-pt) disponibilizado pelo usuário Manuel Romero (mrm8488 do github) foi o que trouxe os melhores resultados.

A Hugging Face é uma startup focada em NLP com uma grande comunidade de código aberto. Eles desenvolveram uma biblioteca (transformers) baseada em python que disponibiliza uma API para as principais arquiteturas conhecidas, como BERT, RoBERTa, GPT-2 ou DistilBERT, que estão sendo utilizadas, com resultados de última geração em uma variedade de tarefas de NLP como: classificação de texto, extração de informações, resposta a perguntas e geração de texto. Essas arquiteturas já possuem diversos corpus pré-treinados em diversas línguas.

