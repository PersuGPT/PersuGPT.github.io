<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> Persuading across Diverse Domains: A Dataset and Persuasion Large Language Model </h1>
<p align='center' style="text-align:center;font-size:1.25em;">
  <a href="https://persugpt.github.io/">Anonymous authors</a><br/>
<sup></sup>Anonymous institutions<br/>
</p>
<p align='center';>
<b>
<em>Under review</em> <br>
</b>
</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://persugpt.github.io/" target="_blank" style="text-decoration: none;">[Paper (Coming soon)]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://persugpt.github.io/" target="_blank" style="text-decoration: none;">[Project Page]</a></br><a href="https://persugpt.github.io/" target="_blank" style="text-decoration: none;">[Dataset (Coming soon)]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://persugpt.github.io/" target="_blank" style="text-decoration: none;">[Demo (Coming soon)]</a>
</b>
</p>

------------

## Introduction

**TL;DR:** We propose a large-scale cross-domain persuasion dataset covers **13,000** scenarios in 35 domains, with the developed PersuGPT model achieving the best performance, surpassing GPT-4 in both automatic and manual evaluation.

<br>

Persuasive dialogue requires multi-turn following and planning abilities to achieve the goal of persuading users, which is still challenging even for state-of-the-art large language models (LLMs). Previous works focus on retrieval-based models or generative models in a specific domain due to a lack of data across multiple domains. In this paper, we leverage GPT-4 to create the first multi-domain persuasive dialogue dataset DailyPersuasion. Then we propose a general method named PersuGPT to learn a persuasion model based on LLMs through intent-to-strategy reasoning, which summarizes the intent of user's utterance and reasons next strategy to respond. Moreover, we design a simulation-based preference optimization, which utilizes a learned user model and our model to simulate next turns and estimate their rewards more accurately. 
Experimental results on two datasets indicate that our proposed method outperforms all baselines in terms of automatic evaluation metric Win-Rate and human evaluation. The code and data are available at [https://persugpt.github.io](https://persugpt.github.io).


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
Coming soon.
```

<br><br>
