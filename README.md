# Quantifying Meibomian Gland Morphology Using Artificial Intelligence

This released repository implements [Quantifying Meibomian Gland Morphology Using Artificial Intelligence](https://journals.lww.com/optvissci/Abstract/2021/09000/Quantifying_Meibomian_Gland_Morphology_Using.15.aspx). More specifically, the gland segmentation of meibography images.

## Requirements
- minoconda3
- pytorch==1.11.0 
- torchvision==0.12.0
- use pip install if you miss any dependencies

## Data and the Trained model
We are not releasing the training data to the public at this moment. However we are providing a sample set that you can run and validate results. See sample set [here](./sample_data). 

To download the trained model, please fill [this form](https://forms.gle/JnzUgCWG34E71rK8A). Note that we will provide you with both upper lid and lower lid segmentation model.

## Cutomized for your own data
If you want to work on your own data, change `data_path` and `raw_dir` in [create_annotations.py](./data/scripts/create_annotations.py) accordingly. 

Then, run the following:
```
python data/scripts/create_annotations.py
```

## Training
We are training upper and lower lid segmentation separately.

Run the following:
```
cd code
python train.py --dataset meibo --gpu YOUR_GPU_NUMBER
```

## Evaluation
Run the following:
```
cd code
python pred_list.py --lst val.lst --usegpu --dataset meibo --model PATH_TO_YOUR_MODEL
```
See some sample visualization results [here](./sample_results/).

## License and Citation
The use of this software is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Please cite our paper if you find this repo useful.
```
@article{wang2021quantifying,
  title={Quantifying meibomian gland morphology using artificial intelligence},
  author={Wang, Jiayun and Li, Shixuan and Yeh, Thao N and Chakraborty, Rudrasis and Graham, Andrew D and Stella, X Yu and Lin, Meng C},
  journal={Optometry and Vision Science},
  volume={98},
  number={9},
  pages={1094--1103},
  year={2021},
  publisher={LWW}
}
```

## Acknowledgement

This repo is heavily based on [instance-segmentation-pytorch](https://github.com/Wizaron/instance-segmentation-pytorch).
