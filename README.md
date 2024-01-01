<div align="center">

<h1>WebVLN: Vision-and-Language Navigation on Websites</h1>

<div>
Qi Chen<sup>*</sup>, Dileepa Pitawela<sup>*</sup>, Chongyang Zhao<sup>*</sup>, Gengze Zhou, Hsiang-Ting Chen, Qi Wu<sup>#</sup>
</div>
Australian Institude for Machine Learning, The University of Adelaide 

<br>

<div>
    <a href='https://github.com/WebVLN/WebVLN' target='_blank'><img alt="Static Badge" src="https://img.shields.io/badge/WebVLN-v1-blue"></a>
    <a href='https://arxiv.org/abs/2312.15820' target='_blank'><img src='https://img.shields.io/badge/Paper-AAAI-green'></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

</div>


## Abstract
Vision-and-Language Navigation (VLN) task aims to enable AI agents to accurately understand and follow natural language instructions to navigate through real-world environments, ultimately reaching specific target locations. We recognise a promising opportunity to extend VLN to a comparable navigation task that holds substantial significance in our daily lives, albeit within the virtual realm: navigating websites on the Internet. This paper proposes a new task named Vision-and-Language Navigation on Websites (WebVLN), where we use question-based instructions to train an agent, emulating how users naturally browse websites. Unlike the existing VLN task that only pays attention to vision and instruction (language), the WebVLN agent further considers underlying web-specific content like HTML, which could not be seen on the rendered web pages yet contains rich visual and textual information. Toward this goal, we contribute a dataset, WebVLN-v1, and introduce a novel approach called Website-aware VLN Network (WebVLN-Net), which is built upon the foundation of state-of-the-art VLN techniques. Experimental results show that WebVLN-Net outperforms current VLN and web-related navigation methods. We believe that the introduction of the new WebVLN task and its dataset will establish a new dimension within the VLN domain and contribute to the broader vision-and-language research community.

<!-- ## Method
![Teaser](hhttps://github.com/WebVLN/WebVLN/method.jpg)
-->

## WebVLN-v1 Dataset
Download the WebVLN-v1 dataset and pre-trained models, and organise data like below:
```
|- WebVLN
    |- Downloads
        |- Data
            |- seen
                | ...
            |- zero_shot
                | ...
            |- img_feats.pkl
            | ...
        |- Oscar
            | ... 
        |- Prevalent
            | ...
```
## WebVLN-Net

### Installation
```bash
cd WebVLN
conda create --name webvlntest python=3.9
conda activate webvlntest
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install nltk putils pytorch_transformers transformers tensorboard tensorboardX networkx
pip install langchain==0.0.246 openai==0.28.1
```
### Training
```bash
cd WebVLN
git checkout main
bash run/train.bash
```

### Influence
```bash
cd WebVLN
git checkout main
bash run/eval.bash
```

### Zero-shot usning LLMs
```bash
cd WebVLN
git checkout zeroshot
python run.py
```


## Citation
If `WebVLN` has been beneficial to your research and work, please cite our work using the following format:
```
@inproceedings{chen2024webvln,
  title={WebVLN: Vision-and-Language Navigation on Websites},
  author={Chen, Qi and Pitawela, Dileepa and Zhao, Chongyang and Zhou, Gengze and Chen, Hsiang-Ting and Wu, Qi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
