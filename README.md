<div align="center">
  <h1>Curriculum-style Local-to-global Adaptation for Cross-domain Remote Sensing Image Segmentation<br></h1>
</div>

<!-- <div align="center">
  <h3><a href=></a>, <a href=></a>, <a href=></a>, <a href=></a></h3>
</div> -->

<div align="center">
  <h4> <a href=https://ieeexplore.ieee.org/document/9576523>[paper] IEEE Link</a></h4>
</div>

<div align="center">
  <h4> <a href=https://ieeexplore.ieee.org/document/9576523>[paper] ArXiv Link</a></h4>
</div>

<div align="center">
  <img src="./figs_tabs/2.png" width=800>
</div>


 <br><br/>
 
If you find our code or paper useful to your research work, please consider citing our work using the following bibtex:
```
@article{zhang2021curriculum,
  title={Curriculum-Style Local-to-Global Adaptation for Cross-Domain Remote Sensing Image Segmentation},
  author={Zhang, Bo and Chen, Tao and Wang, Bin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={60},
  pages={1--12},
  year={2021},
  publisher={IEEE}
}
```

## Preprocessing data
Following [DualGAN](https://www.sciencedirect.com/science/article/pii/S0924271621000423), we crop the whole images in Potsdam IR-R-G dataset into the size of 512 × 512 with both horizontal and vertical strides of 512 pixels, and generate 4598 patches. For Vaihingen dataset, we crop the whole images into a size of 512 × 512 with both horizontal and vertical strides of 256 pixels and obtain 1696 patches

The following processed datasets are used in our paper: 
- PotsdamIRRG \[[Dataset Page](https://drive.google.com/file/d/1EuTBY25cq65KBYfCcCkcMqB0pMOQHGNw/view?usp=sharing)\]
- PotsdamRGB \[[Dataset Page](https://drive.google.com/file/d/1EuTBY25cq65KBYfCcCkcMqB0pMOQHGNw/view?usp=sharing)\]
- Vaihingen \[[Dataset Page](https://drive.google.com/file/d/1EuTBY25cq65KBYfCcCkcMqB0pMOQHGNw/view?usp=sharing)\]

After dowloading datasets, copy the data.zip to /ADVENT/. and extract it:
```
unzip data.zip
```

## Dowloading the ImageNet pretrained model
- ImageNet pretrained weights \[[Dataset Page](https://drive.google.com/file/d/1CZIJ5IJMPrsFB5URU76GJ5LjWqpRKjSM/view?usp=sharing)\]

## Train the source-only model from ImageNet pretrained model
```
cd /SegNet_Source/ADVENT/.
pip install -e .
cd /SegNet_Source/ADVENT/scripts/.
python train.py --cfg /root/code/SegNet_Source/ADVENT/advent/scripts/configs/advent.yml
```

## Test the source-only model
```
python test.py --cfg /root/code/SegNet_Source/ADVENT/advent/scripts/configs/advent.yml
```

## Contact
We have tried our best to verify the correctness of our released data, code and trained model weights. 
However, there are a large number of experiment settings, all of which have been extracted and reorganized from our original codebase. 
There may be some undetected bugs or errors in the current release. 
If you encounter any issues or have questions about using this code, please feel free to contact us via bo.zhangzx@gmail.com

