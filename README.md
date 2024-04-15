
<p align="center">
<img src="https://github.com/Uniaa-MLLM/Uniaa/blob/main/imgs/uniaa.png" width="10%">
</p>

 <div>
<a href="https://github.com/Uniaa-MLLM/Uniaa"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FUniaa-MLLM%2FUniaa&count_bg=%23E97EBA&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false"/></a>    
<a href="https://github.com/Uniaa-MLLM/Uniaa"><img src="https://img.shields.io/github/stars/Uniaa"/></a>
<a href="https://arxiv.org/abs/2309.14181"><img src="https://img.shields.io/badge/Arxiv-2309:14181-red"/></a>
<a href="https://github.com/Q-Future/Q-Bench/releases/tag/v1.0.1.1014datarelease"><img src="https://img.shields.io/badge/Data-Release-green"></a>
<!-- <a href="https://arxiv.org/abs/2402.07116"><img src="https://img.shields.io/badge/Extension-2402:07116-yellow"/></a> -->
<!-- <a href="https://github.com/Q-Future/Q-Instruct"><img src="https://img.shields.io/badge/Awesome-QInstruct-orange"/></a> -->
   </div>

# Uniaa: A Unified Multi-modal Image Aesthetic Assessment Baseline and Benchmark

The Unified Multi-modal Image Aesthetic Assessment Framework, containing a baseline (a) and a benchmark  (b). The aesthetic perception performance of UNIAA-LLaVA and other MLLMs is shown in (c).
<p align="center">
      <img style="width:80%" src="imgs/intro.png">
</p>



    
The IAA Datasets Conversion Paradigm for UNIAA-LLaVA.
<p align="center">
      <img style="width:80%" src="imgs/baseline.png">
</p>


The UNIAA-Bench overview. (a) UNIAA-QA contains 5354 Image-Question-Answer samples and (b) UNIAA-Describe contains 501 Image-Description samples. (c) For open-source MLLMs, Logits can be extracted to calculate the score.
<p align="center">
      <img style="width:80%" src="imgs/benchmark.png">
</p>




## Release

- [4/15] 🔥 We build the page of UNIAA!
  

## Performance

### Aesthetic Perception Performance

  <img style="width:80%" src="imgs/perception.png">


### Aesthetic Description Performance

   <img style="width:40%" src="imgs/description.png">


### Aesthetic Assessment Performance
#### Zero-shot

 <img style="width:40%" src="imgs/zero-shot-assessment.png">


#### Supervised learning on AVA and TAD66K

  <img style="width:40%" src="imgs/superivised-learning-assessment.png">



## Training on data of UNIAA
#### Step 1: Download Images and Json files
#### Step 2: Training On Specific MLLM

## Test on UNIAA-Bench
### For Aesthetic Perception
#### Step 1: Download Images and Json files
#### Step 2: Run the inference code
#### Step 3: Calculate the score

### For Aesthetic Description
#### Step 1: Download Images and Json files
#### Step 2: Run the inference code
