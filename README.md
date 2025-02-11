<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Persuading across Diverse Domains: A Dataset and Persuasion Large Language Model </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
Chuhao Jin, Kening Ren, Lingzhen Kong, Xiting Wang, Ruihua Song, Huan Chen<br/>
<sup></sup>Renmin Universuty of China, Meituan<br/>
<b>
<em>ACL 2024</em> <br>
</b>
</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://aclanthology.org/2024.acl-long.92.pdf" target="_blank" style="text-decoration: none;">[Paper]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://persugpt.github.io/" target="_blank" style="text-decoration: none;">[Project Page]</a></br><a href="https://persugpt.github.io/" target="_blank" style="text-decoration: none;">[Dataset]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="http://persugpt.mmd.ac.cn:7860/" target="_blank" style="text-decoration: none;">[Demo]</a>
</b>
</p>


**TL;DR:** We propose a large-scale cross-domain persuasion dataset covers **13,000** scenarios in 35 domains, with the developed PersuGPT model achieving the best performance, surpassing GPT-4 in both automatic and manual evaluation.


## Abstract
Persuasive dialogue requires multi-turn following and planning abilities to achieve the goal of persuading users, which is still challenging even for state-of-the-art large language models (LLMs). Previous works focus on retrieval-based models or generative models in a specific domain due to a lack of data across multiple domains. In this paper, we leverage GPT-4 to create the first multi-domain persuasive dialogue dataset DailyPersuasion. Then we propose a general method named PersuGPT to learn a persuasion model based on LLMs through intent-to-strategy reasoning, which summarizes the intent of user's utterance and reasons next strategy to respond. Moreover, we design a simulation-based preference optimization, which utilizes a learned user model and our model to simulate next turns and estimate their rewards more accurately. 
Experimental results on two datasets indicate that our proposed method outperforms all baselines in terms of automatic evaluation metric Win-Rate and human evaluation. The code and data are available at [https://persugpt.github.io](https://persugpt.github.io).


## ***News***

- *02/11/2025* 🌟 We release the PersuGPT model and inference code, as well as an online [demo](http://persugpt.mmd.ac.cn:7860/)!
- *05/16/2024* 🎉 Our DailyPersuasion and PersuGPT are accepted by ACL 2024!
- *03/29/2024* 🌟 We release the English version of the DailyPersuasion dataset!

## Download
You can download the checkpoint of PersuGPT from [here](https://huggingface.co/Chuhaojin/PersuGPT).


## Inference
Please move the checkpoint to ```ckpt```, and run the following command:
```shell
bash web_demo.sh
```

You can also run the following command for cli demo:
```shell
bash cli_demo.sh
```

## Requirements

```
transformers>=4.36.2
datasets>=2.14.3
accelerate>=0.21.0
peft>=0.7.0
trl>=0.7.6
gradio>=3.38.0,<4.0.0
scipy
sentencepiece
protobuf
tiktoken
jieba
rouge-chinese
nltk
uvicorn
pydantic
fastapi
sse-starlette
matplotlib
```


## A Quick Glance
<p align="center">
<img src="./website/images/main_fig.png" alt="main_figure" width="90%"/>
</p>
<p align="center">
<font size=3 >Quick Glance of Proposed DailyPersuasion Dataset.</font>
</p>

<br>


## Ethics and Disclosure
Persuasive dialogue systems serve as a double-edged sword. On one hand, they can be extensively applied in psychological therapy and philanthropic efforts, fostering positive developments within human society. On the other hand, their misuse in potentially harmful scenarios must be strictly regulated. In our study, we filter the keywords used to construct persuasive scenarios, ensuring all generated scenarios are safe and free from bias. We utilize GPT-4, aligned with ethical values, to collect data, hoping to guarantee the gathered data is devoid of user privacy breaches and harmful content as GPT-4. We will ask humans to review all scenarios, dialogues, and strategies before releasing DailyPersuasion and further filter inappropriate or risky data. We will also ask all people or organizations that download the dataset to sign a strict license to manage the use of our data. It is worth noting that, while our system can be employed across various persuasive domains, it should not be used to directly replace human interaction. All applications of our system should operate under human supervision and regulation, maintaining a balance between leveraging technology for good and ensuring ethical use.


## Citation
If you find this useful in your research, please consider citing:

```
@inproceedings{jin-etal-2024-persuading,
    title = "Persuading across Diverse Domains: a Dataset and Persuasion Large Language Model",
    author = "Jin, Chuhao  and
      Ren, Kening  and
      Kong, Lingzhen  and
      Wang, Xiting  and
      Song, Ruihua  and
      Chen, Huan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.92",
    pages = "1678--1706",
    abstract = "Persuasive dialogue requires multi-turn following and planning abilities to achieve the goal of persuading users, which is still challenging even for state-of-the-art large language models (LLMs). Previous works focus on retrieval-based models or generative models in a specific domain due to a lack of data across multiple domains. In this paper, we leverage GPT-4 to create the first multi-domain persuasive dialogue dataset DailyPersuasion. Then we propose a general method named PersuGPT to learn a persuasion model based on LLMs through intent-to-strategy reasoning, which summarizes the intent of user{'}s utterance and reasons next strategy to respond. Moreover, we design a simulation-based preference optimization, which utilizes a learned user model and our model to simulate next turns and estimate their rewards more accurately. Experimental results on two datasets indicate that our proposed method outperforms all baselines in terms of automatic evaluation metric Win-Rate and human evaluation. The code and data are available at https://persugpt.github.io.",
}
```

<br><br>
