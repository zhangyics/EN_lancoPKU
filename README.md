
## Machine Learning

- ### [meProp](https://github.com/lancopku/meProp)

  Code for “meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting”[[pdf]](http://proceedings.mlr.press/v70/sun17c/sun17c.pdf). This work only computes a small subset of the full gradient to update the model parameters in back propagation, leading to a linear reduction in the computational cost. This does not result in a larger number of training iterations. More interestingly, the accuracy of the resulting models is actually improved.  
  论文 “meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting”[[pdf]](http://proceedings.mlr.press/v70/sun17c/sun17c.pdf)相关代码。此项工作在后向传播中仅使用一小部分梯度来更新模型参数，计算成本呈线性减少，模型训练轮数并未增加，而准确度却有所提高。


- ### [meSimp](https://github.com/lancopku/meSimp)

  Codes for “Training Simplification and Model Simplification for Deep Learning: A Minimal Effort Back Propagation Method" [[pdf]](https://arxiv.org/pdf/1711.06528.pdf). This work only computes a small subset of the full gradient to update the model parameters in back propagation and further simplify the model by eliminating the rows or columns that are seldom updated. Experiments show that the model could often be reduced by around 9x, without any loss on accuracy or even with improved accuracy.   
  论文 "Training Simplification and Model Simplification for Deep Learning: A Minimal Effort Back Propagation Method" [[pdf]](https://arxiv.org/pdf/1711.06528.pdf)相关代码。此项工作在后向传播中仅使用一小部分梯度更新参数并消除了参数矩阵中一些很少被更新到的行或列。实验显示模型通常可被简化9倍左右，准确度并未受损甚至有所上升。


- ### [Label embedding](https://github.com/lancopku/label-embedding-network)

  Code for “paper Label Embedding Network: Learning Label Representation for Soft Training of Deep Networks”[[pdf]](https://arxiv.org/pdf/1710.10393.pdf). This work learns label representations and makes the originally unrelated labels have continuous interactions with each other during the training process. The trained model can achieve substantially higher accuracy and with faster convergence speed. Meanwhile, the learned label embedding is reasonable and interpretable.  
  论文 “paper Label Embedding Network: Learning Label Representation for Soft Training of Deep Networks”[[pdf]](https://arxiv.org/pdf/1710.10393.pdf)相关代码。这项工作在训练过程中学习标签表示，并让以往并无关联的标签彼此之间产生了交互。模型收敛加快且极大地提高了准确度。同时，学习到的标签表示也更具合理性与可解释性。


## Machine Translation

- ### [Deconv Dec](https://github.com/lancopku/DeconvDec)

  Code for “Deconvolution-Based Global Decoding for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1806.03692.pdf). This work proposes a new NMT model that decodes the
sequence with the guidance of its structural prediction of the context of the target sequence. The model gets very competitive results. It is robust to translating sentences of different lengths and it also
reduces repetition repetition phenomenon.  
  论文"Deconvolution-Based Global Decoding for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1806.03692.pdf)相关代码。这项工作提出了一个新的神经机器翻译模型，以对目标序列上下文的结构预测为指导来生成序列，模型获得了极具竞争力的结果，对于不同长度的序列鲁棒性更强，且减轻了生成序列中的重复现象。

- ### [bag-of-words](https://github.com/lancopku/bag-of-words)

  Code for “Bag-of-Words as Target for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1805.04871.pdf). This work uses both the sentences and the bag-of-words as targets in the training stage, which encourages
the model to generate the potentially correct sentences that are not appeared in the training set. Experiments show the model outperforms the strong baselines by a large margin.
  论文“Bag-of-Words as Target for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1805.04871.pdf)相关代码。这项工作将目标语句与目标的词袋都作为训练目标，使得模型能够生成出有可能正确却不在训练集中的句子。实验显示模型BLEU值大幅优于基线模型。

- ### [ACA4NMT](https://github.com/lancopku/ACA4NMT)

  Code for “Decoding History Based Adaptive Control of Attention for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1802.01812.pdf). This model learns to control the attention by
keeping track of the decoding history. The model is capable of generating translation with less repetition
and higher accuracy.     
  论文“Decoding History Based Adaptive Control of Attention for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1802.01812.pdf)相关代码。该模型通过追踪解码历史来控制注意力机制。模型能够较少重复地生成翻译且精度更高。



## Summarization 


- ### [LancoSum](https://github.com/lancopku/LancoSum) (toolkit)
  This repository provides a toolkit for abstractive summarization, which can assist researchers to implement the common baseline, the attention-based sequence-to-sequence model, as well as three high quality models proposed by our group LancoPKU recently. By modifying the configuration file or the command options, one can easily apply the models to his own work.   
  此项目提供了一个针对生成式摘要的工具包，包含通用的基线模型——基于注意力机制的序列到序列模型以及LancoPKU组近期提出的三个高质量摘要模型。通过修改配置文件或命令行，研究者可方便地将其应用至自己的工作。

- ### [Global-Encoding](https://github.com/lancopku/Global-Encoding)

  Code for “Global Encoding for Abstractive Summarization” [[pdf]](https://arxiv.org/abs/1805.03989). This work proposes a framework which controls the information flow from the encoder to the decoder based on the global information of the source context. The model outperforms the baseline models and is capable of reducing repetition.    
  论文“Global Encoding for Abstractive Summarization” [[pdf]](https://arxiv.org/abs/1805.03989)相关代码。这项工作提出了一个基于全局源语言信息控制编码段到解码端信息流的框架，模型优于多个基线模型且能够减少重复输出。

- ### [HSSC](https://github.com/lancopku/HSSC)

  Code for “A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification”[[pdf]](https://arxiv.org/pdf/1805.01089.pdf). This work proposes a model for joint learning of text summarization and sentiment classification. Experimental results show that the proposed model achieves better performance than the strong baseline systems on both abstractive summarization and sentiment classification.  
  论文“A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification”[[pdf]](https://arxiv.org/pdf/1805.01089.pdf)相关代码。这项工作提出了一个联合学习摘要和情感分类的任务的模型。实验显示所提出模型在两项任务上都取得了相较强基线模型更好的效果。


- ### [WEAN](https://github.com/lancopku/WEAN)

  Code for “Query and Output: Generating Words by Querying Distributed Word Representations for Paraphrase Generation”[[pdf]](https://arxiv.org/pdf/1803.01465.pdf). This model generates the words by querying
distributed word representations (i.e. neural word embeddings) in summarization. The model outperforms the baseline models by a large margin and achieves state-of-the-art performances
on three benchmark datasets.  
  论文“Query and Output: Generating Words by Querying Distributed Word Representations for Paraphrase Generation”[[pdf]](https://arxiv.org/pdf/1803.01465.pdf)相关代码。在生成摘要时，该模型通过查询单词表示（词嵌入）来产生新的单词。模型大幅度优于基线模型且在三个基准数据集上达到了最优表现。


- ### [SRB](https://github.com/lancopku/SRB)

  Code for “Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization”[[pdf]](https://arxiv.org/pdf/1706.02459.pdf).  This work improves the semantic
relevance between source texts and summaries in Chinese social media summarization by encouraging
high similarity between the representations of texts and summaries.    
  论文“Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization”[[pdf]](https://arxiv.org/pdf/1706.02459.pdf)相关代码。这项工作使源文本与摘要的表示获得尽可能高的相似度，而极大地提高了源文本与生成摘要的语义关联。




- ### [superAE](https://github.com/lancopku/superAE)

  Code for “Autoencoder as Assistant Supervisor: Improving Text Representation for Chinese Social Media Text Summarization”[[pdf]](https://arxiv.org/pdf/1805.04869.pdf). This work regard a summary autoencoder as an assistant supervisor of Seq2Seq to get more informative representation of source content. Experimental results show that the model achieves the state-of-the-art performances on the benchmark dataset.   
  论文“Autoencoder as Assistant Supervisor: Improving Text Representation for Chinese Social Media Text Summarization”[[pdf]](https://arxiv.org/pdf/1805.04869.pdf)相关代码。这项工作将摘要自编码器作为给序列到序列模型的一个监督信号，来获得更具信息量的源文本表示。实验结果显示模型在基准数据集上获得了最优效果。


## Text Generation

- ### [Unpaired-Sentiment-Translation](https://github.com/lancopku/Unpaired-Sentiment-Translation)

  Code for “Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach" [[pdf]](https://arxiv.org/abs/1805.05181). This work proposes a cycled reinforcement learning method to realize sentiment-to-sentiment translation. The proposed method does not rely on parallel data and significantly outperforms the state-of-the-art systems in terms of the content preservation.        
  论文“Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach" [[pdf]](https://arxiv.org/abs/1805.05181)相关代码。这项工作提出了一个循环强化学习的方式来实现情感转换。所提出方法不依赖任何平行语料，且在内容保留程度上，显著地优于现有的最优模型。



- ### [DPGAN](https://github.com/lancopku/DPGAN)

  Code for “DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text” [[pdf]](https://arxiv.org/pdf/1802.01345.pdf). This work novelly introduces a language-model based discriminator in Generative Adversarial Network. The proposed model can generate substantially more diverse and informative text than existing baseline methods.    
  论文“DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text” [[pdf]](https://arxiv.org/pdf/1802.01345.pdf)相关代码。这项工作在生成对抗网络中创新地引入了一个基于语言模型的判别器。所提出模型能够生成显著得优于基线方法的更具多样性和信息量的文本。


## Dependency Parsing

- ### [nndep](https://github.com/lancopku/nndep)

  Code for “Hybrid Oracle: Making Use of Ambiguity in Transition-based Chinese Dependency Parsing”[[pdf]](https://arxiv.org/pdf/1711.10163.pdf).   
  论文“Hybrid Oracle: Making Use of Ambiguity in Transition-based Chinese Dependency Parsing”[[pdf]](https://arxiv.org/pdf/1711.10163.pdf)相关代码。

## Sequence Labeling

- ### [PKUSeg](https://github.com/lancopku/PKUSeg) (toolkit)

  This repository provides a toolkit for Chinese segmentation.  
  此项目提供了一个针对中文分词的工具包。PKUSeg简单易用，支持多领域分词，在不同领域的数据上都大幅提高了分词的准确率。

- ### [ChineseNER](https://github.com/lancopku/ChineseNER)

  Code for “Cross-Domain and Semi-Supervised Named Entity Recognition in Chinese Social Media: A Unified Model”  
  论文“Cross-Domain and Semi-Supervised Named Entity Recognition in Chinese Social Media: A Unified Model”相关代码。
  

- ### [Multi-Order-LSTM](https://github.com/lancopku/Multi-Order-LSTM)

  Code for “Does Higher Order LSTM Have Better Accuracy for Segmenting and Labeling Sequence Data?”[[pdf]](https://arxiv.org/pdf/1711.08231.pdf). This work combines low order and high order LSTMs together and considers longer distance dependencies of tags into consideration. The model is scalable to higher order models and especially performs well in recognizing long entities.   
  论文"Does Higher Order LSTM Have Better Accuracy for Segmenting and Labeling Sequence Data?”[[pdf]](https://arxiv.org/pdf/1711.08231.pdf)相关代码。此项工作合并了低阶LSTM模型和高阶LSTM模型，考虑到了标签之间的长距离依赖。模型保持了对更高阶的模型的扩展性，尤其对于长实体的识别表现优异。


- ### [Decode-CRF](https://github.com/lancopku/Decode-CRF)

  Code for “Conditional Random Fields with Decode-based Learning: Simpler and Faster”[[pdf]](https://arxiv.org/pdf/1503.08381.pdf). This work proposes a decode-based probabilistic online learning method, This method is with fast training, very simple to implement, with top accuracy, and with theoretical guarantees of convergence.  
  论文“Conditional Random Fields with Decode-based Learning: Simpler and Faster”[[pdf]](https://arxiv.org/pdf/1503.08381.pdf)相关代码。此项工作提出了一个基于解码的概率化在线学习方法。该方法训练很快，易于实现，准确率高，且理论可收敛。


- ### [SAPO](https://github.com/lancopku/SAPO)

  Code for “Towards Shockingly Easy Structured Classification: A Search-based Probabilistic Online Learning Framework”[[pdf]](https://arxiv.org/pdf/1503.08381.pdf).    
  论文”Towards Shockingly Easy Structured Classification: A Search-based Probabilistic Online Learning Framework”[[pdf]](https://arxiv.org/pdf/1503.08381.pdf)相关代码。


## Text Classification

- ###  [SGM](https://github.com/lancopku/SGM)

  Code for “SGM: Sequence Generation Model for Multi-label Classification”[[pdf]](https://arxiv.org/pdf/1806.04822.pdf). This work views the multi-label classification task as a sequence generation
problem. The proposed methods not only capture the correlations between labels, but also select the most informative
words automatically when predicting different labels.    
  论文“SGM: Sequence Generation Model for Multi-label Classification”[[pdf]](https://arxiv.org/pdf/1806.04822.pdf)相关代码。此项工作将多标签分类任务看做序列生成任务。所提出方法不仅能捕捉标签之间的关联，还能在预测不同标签时自动选择出最具信息量的单词。



## Applied Tasks

- ### [AAPR](https://github.com/lancopku/AAPR)

  Code for “Automatic Academic Paper Rating Based on Modularized Hierarchical Convolutional Neural Network”
[[pdf]](https://arxiv.org/pdf/1805.03977.pdf). This work builds a
new dataset for automatically evaluating academic papers and propose a novel modularized hierarchical convolutional neural network to for this task.  
  论文“Automatic Academic Paper Rating Based on Modularized Hierarchical Convolutional Neural Network”
[[pdf]](https://arxiv.org/pdf/1805.03977.pdf)相关代码。此项工作建立了一个自动评估学术论文的的数据集，并提出了一个适用用此任务的模块化的层级卷积网络。


- ### [tcm_prescription_generation](https://github.com/lancopku/tcm_prescription_generation)

  Code for “Exploration on Generating Traditional ChineseMedicine Prescriptions from Symptoms with an End-to-End Approach”[[pdf]](https://arxiv.org/pdf/1801.09030.pdf). This work explores the Traditional Chinese Medicine prescription generation task using seq2seq models.   
  论文“Exploration on Generating Traditional ChineseMedicine Prescriptions from Symptoms with an End-to-End Approach”[[pdf]](https://arxiv.org/pdf/1801.09030.pdf)相关代码。此项工作利用序列到序列模型，探索了传统中医的药方生成任务。



## Datasets

- ### [Chinese-Literature-NER-RE-Dataset](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)

  Data for “A Discourse-Level Named Entity Recognition and Relation Extraction Dataset for Chinese Literature Text” [[pdf]](https://arxiv.org/pdf/1711.07010.pdf). This work builds a discourse-level dataset from hundreds of Chinese literature articles for improving Named Entity Recognition and Relation Extraction for Chinese literature text.    
  论文“A Discourse-Level Named Entity Recognition and Relation Extraction Dataset for Chinese Literature Text” [[pdf]](https://arxiv.org/pdf/1711.07010.pdf)相关数据。本篇工作从数百篇中文散文中建立了一个篇章级别的数据集，旨在提高命名实体识别和关系抽取任务在散文上的表现。

- ### [Chinese-Dependency-Treebank-with-Ellipsis](https://github.com/lancopku/Chinese-Dependency-Treebank-with-Ellipsis)

  Data for “Building an Ellipsis-aware Chinese Dependency Treebank for Web Text”[[pdf]](https://arxiv.org/pdf/1801.06613.pdf). This work builds a Chinese weibo dependency treebank which contains 572
sentences with omissions restored and contexts reserved, aimed at improving dependency parsing for texts with ellipsis.    
  论文“Building an Ellipsis-aware Chinese Dependency Treebank for Web Text”[[pdf]](https://arxiv.org/pdf/1801.06613.pdf)相关数据。本篇工作建立了一个中文微博依存树库，包含572个在保留语义的情况下还原了省略语的句子，旨在提高依存句法分析在存在省略的文本上的表现。

- ### [Chinese-abbreviation-dataset](https://github.com/lancopku/Chinese-abbreviation-dataset)

  Data for “A Chinese Dataset with Negative Full Forms for General Abbreviation Prediction” [[pdf]](https://arxiv.org/pdf/1712.06289.pdf). This work builds a dataset for general Chinese abbreviation prediction. The dataset incorporates negative full forms to promote the research in this area.  
  论文“A Chinese Dataset with Negative Full Forms for General Abbreviation Prediction” [[pdf]](https://arxiv.org/pdf/1712.06289.pdf)相关数据。本篇工作建立了一个通用的中文缩略语预测数据集，该数据集涵盖了无缩略语的完全短语，旨在促进这一领域的研究。




