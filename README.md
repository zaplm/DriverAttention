# Towards Robust Unsupervised Attention Prediction in Autonomous Driving

This repository is an extension of our ICCV conference paper, **"Unsupervised Self-Driving Attention Prediction via Uncertainty Mining and Knowledge Embedding."** For the original ICCV code, please check out the `iccv` branch.

## Dataset Preparation

The DriverAttention-C dataset can be downloaded from [this link](https://drive.google.com/file/d/1p9rmy3dXESSaHiGHApxcy-aQlAymDzz7/view?usp=sharing).

The dataset is organized as follows:
```
root/
├── scene_0/
│   ├── camera
│   ├── camera_224_224
│   ├── gaze
│   ├── gaze_224_224
│   ├── gaussian_noise_224_224
│   ├── ...
│   ├── fog
│   └── snow 
├── scene_1/
├── ...
├── scene_n/
├── train.txt
├── val.txt
├── test_cor.txt
├── hard_cases_gl2.0.txt
├── ...
└── hard_cases_gl4.0.txt
```
The `hard_cases_gl*.txt` files can be found in the `lt_txts/` folder.

## Corruption Robustness

**Training:**
```bash
python train_robo_cor.py --name exp_name --data-path path/to/data --topK 8 --mix_dir temp_dir
```

**Evaluation:**
```bash
python test_cor.py --data-path path/to/data --save_model model_name
```

## Central Bias

**Training:**
```bash
python train_longtail.py --name rcpreg --data-path path/to/data --batch-size 4
```

**Evaluation:**
```bash
python test_longtail.py --data-path path/to/data --save_model save_weights
```

## Decision-Making

**Training:**
1. Prepare the data as described in [this repository](https://github.com/Twizwei/bddoia_project).
2. Use the trained model to infer driver attention.
3. Train the decision-making model:
```
python train_decision.py --name test_ --atten_model {infer_dir} --data-path path/to/data
```

**Evaluation:**
Follow the instructions in [the repository](https://github.com/Twizwei/bddoia_project).


## Citing

If our work proves helpful in your research, please acknowledge it by citing the following BibTeX entry:

```bibtex
@article{qi2025towards,
  title={Towards Robust Unsupervised Attention Prediction in Autonomous Driving},
  author={Qi, Mengshi and Bi, Xiaoyang and Zhu, Pengfei and Ma, Huadong},
  journal={arXiv preprint arXiv:2501.15045},
  year={2025}
}
```
