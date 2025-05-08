# dataset preparetion
the DriverAttention-C can be downloaded at https://drive.google.com/file/d/1W8joMa6blxwNrETF2005cCFViq4DCNvB/view?usp=sharing.
each dataset's structure is organized as follows:
```
root/
├── scene_0/
│   ├── camera
│   ├── camera_224_224
│   └── gaze
│   └── gaze_224__224
│   └── gaussian_noise_224_224 
├── scene_1/
├── .....
├── scene_n/
├── train.txt
├── val.txt
├── test_cor.txt
├── hard_cases_gl2.0.txt
├── ....
└── hard_cases_gl4.0.txt
```
the hard_cases_gl*.txt can be found at lt_txts/ folder

# train corruption robustness
```bash
python train_robo_cor.py --name exp_name --data-path path/to/data --topK 8 --mix_dir temp_dir
```


# eval corruption robustness
```bash
python test_cor.py --data-path ../atten_data/bd --save_model bd
```


# train for central bias
```bash
python train_longtail.py --name rcpreg --data-path ../atten_data/bd --batch-size 4
```


# test for cental bias
```bash
python test_longtail.py --data-path /path/to/data --save_model save_weights
```
