# GPT-3: Language Models are Few-Shot Learners

[arXiv link](https://arxiv.org/abs/2005.14165)
> Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting.  For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model.  GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans.  We discuss broader societal impacts of this finding and of GPT-3 in general.

>(Translate) 최근 연구는 대규모 텍스트 자료의 사전학습을 실시한 후 특정 Task에 대해 정밀 조정을 통해 많은 자연어처리 과제와 벤치마크에서 상당한 소득을 얻고 있음을 증명했다.
 일반적으로 아키텍쳐에서는 Task에 구애받지 않으나, 이 방법은 여전히 특정 작업별, 잘 튜닝된 방대한 양의 샘플 데이터셋을 필요로 한다. 대조적으로, 사람들은
 일반적으로 단지 몇개의 예제 또는 간단한 지침으로부터 언어 과제를 수행할 수 있다. 이는 현재의 자연어처리 시스템이 아직도 대부분 수행하기 힘든 부분이다. 
 여기서 우리는 언어모델을 확장함으로써 Task에 구애받지 않고, 적은 데이터로 성능이 개선되며, 때로는 이전 최첨단 미세조정 수준까지의 경쟁력에 도달할 수 있음을 보여준다.
 특히, 우리는 이전에 non-sparse 언어모델보다 10배 더 많은 1750억개의 파라미터들을 가지고 있는 자동회귀 언어모델인 GPT-3를 학습시키고, few-shot setting에서 그 모델의 성능을 테스트했다.
 모든 Task에 대해서, GPT-3는 Tasks 그리고 모델과의 문자 상호작용을 통해 순수하게 특정된 Few-shot 증명을 수행하면서 어떠한 Gradient 업데이트나 미세조정없이 적용되었다.
 GPT-3는 번역, 질의-응답, 빈칸메우기 뿐만아니라 문장에서 참신한 단어를 사용하거나 3자리 산수를 수행하는 등의 Domain 적응이나 즉석추론이 필요한 몇몇 작업 등 많은 자연어처리 데이터셋에서 
 강력한 성능을 보여주었다.
 동시에, GPT-3의 Few-shot learning이 여전히 힘든 몇몇 데이터셋과, 거대한 웹자료 학습과 연관된 방법론적 문제들을 확인했다. 
 마침내 우리는 GPT-3가 사람이쓴 기사인지 아닌지 분별하기 힘들정도의 뉴스기사 샘플을 생성할 수 있다는 것을 확인했다. 
 
 
## Contents
- [175b_samples.jsonl](175b_samples.jsonl) - Unconditional, unfiltered 2048 token samples from GPT-3 with p=.85, t=1.&#12288;
**CONTENT WARNING:** GPT-3 was trained on arbitrary data from the web, so may contain offensive content and language.
- [data](data) - Synthetic datasets for word scramble and arithmetic tasks described in the paper.
- [dataset_statistics](dataset_statistics) - Statistics for all languages included in the training dataset mix.
- [overlap_frequency.md](overlap_frequency.md) - Samples of 13-gram overlaps between our training data and benchmarks, selected by frequency in the training set.


## How to cite
```
@article{brown2020language,
    title={Language Models are Few-Shot Learners},
    author={Tom B. Brown and Benjamin Mann and Nick Ryder and Melanie Subbiah and Jared Kaplan and Prafulla Dhariwal and Arvind Neelakantan and Pranav Shyam and Girish Sastry and Amanda Askell and Sandhini Agarwal and Ariel Herbert-Voss and Gretchen Krueger and Tom Henighan and Rewon Child and Aditya Ramesh and Daniel M. Ziegler and Jeffrey Wu and Clemens Winter and Christopher Hesse and Mark Chen and Eric Sigler and Mateusz Litwin and Scott Gray and Benjamin Chess and Jack Clark and Christopher Berner and Sam McCandlish and Alec Radford and Ilya Sutskever and Dario Amodei},
    year={2020},
    eprint={2005.14165},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
