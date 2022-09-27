# BGGunSoundDataset
This repository provides the official BGG dataset and implementation of the following paper: 
"Enemy Spotted: In-game Gun Sound Dataset for Gunshot Classification and Localization" accepted at IEEE Conference on Games (GoG) 2022.

The BGG dataset is collected from the game, "PLAYERUNKNOWN'S BATTLEGROUNDS" (PUBG) made by Crafton, Inc. We use the in-game sounds for **non-commercial** research purposes only. 

![demo](https://user-images.githubusercontent.com/26558158/183609029-9fa05f22-4adb-4c67-993b-60fd1c4c1029.jpg)

## Env
```
python 3.8
torch 1.7
librosa
```

## Dataset
1. Unzip the "[gun_sound_v2.zip](https://drive.google.com/file/d/1TIEgt1KEJtcK5zDhnuHvtK1nisv-C1fj/view?usp=sharing)" in data folder.
2. Follow the path structure below.
```
BGGunSoundDataset
├── data
    ├── gun_sound_v2
    │   ├── ak_0m_center_0000.mp3
    │   ├── ak_0m_center_0001.mp3
    │   ├──  ...
    │   └── scar_0m_center_2194.mp3
    ├── v3_exp1_train.csv
    ├── v3_exp1_test.csv
    ├── v3_exp2_train.csv
    └── v3_exp2_test.csv
```

## Run Gunshot classfication and Localization on BGG dataset.

### Gunshot Classification
```
train_classification.py  --backbone {CNN, RNN, CRNN, Trans, CTrans} --lr 1e-3 --input_sec 2
```
### Gunshot Localization
```
train_localization_and_classification.py --backbone {CNN, RNN, CRNN, Trans, CTrans} --lr 1e-4 --input_sec 3
```

## Citation
```
@inproceedings{park2022enemy,
  title={Enemy Spotted: In-game Gun Sound Dataset for Gunshot Classification and Localization},
  author={Park, Junwoo and Cho, Youngwoo and Sim, Gyuhyeon and Lee, Hojoon and Choo, Jaegul},
  booktitle={2022 IEEE Conference on Games (CoG)},
  pages={56--63},
  year={2022},
  organization={IEEE}
}
```
