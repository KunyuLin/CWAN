# Adversarial Partial Domain Adaptation by Cycle Inconsistency
PyTorch implementation of the ECCV2022 paper “Adversarial Partial Domain Adaptation by Cycle Inconsistency” 
[[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1819_ECCV_2022_paper.php)

## Environments
- Python: 3.7.3
- PyTorch: 1.8.1

## Training
#### Data preparation
By running the script `examples/domain_adaptation/partial/run.sh`, datasets will be downloaded automatically to the directory `examples/domain_adaptation/partial/data/`.

Or you can download the datasets and put them into the right directory. 


#### Run training scripts
```
cd examples/domain_adaptation/partial/
bash run.sh
```

## Acknowledgement 
- This repository is heavily based on the codebase [thuml/Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library). 

- If you find this paper/code useful, please consider citing us:
```
@inproceedings{lin2022partial,
  author       = {Kun-Yu Lin and Jiaming Zhou and Yukun Qiu and Wei-Shi Zheng},
  title        = {Adversarial Partial Domain Adaptation by Cycle Inconsistency},
  booktitle    = {European Conference on Computer Vision},
  year         = {2022},
}
```
