# LiDAR_based_APR
Unofficial implementation of LiDAR-based Absolute Pose Regression models

## Model Stats
| **Model** | **Params (M)** | **FLOPs (G)** | **Time (ms)** | **Weight (MB)** | **pth** |
|:----------|:--------------:|:-------------:|:-------------:|:---------------:|:-------:|
| PointLoc  |     3.321      |     5.248     |    139.319    |     12.756      |         |
| PosePN    |     1.460      |     0.592     |     0.106     |      5.606      |         |
| PosePNPP  |     5.878      |    11.330     |    135.992    |     22.542      |         |
| PoseSOE   |     5.254      |     3.165     |    27.067     |     20.148      |         |
| STCLoc    |     9.217      |     0.258     |    23.478     |     35.294      |         |
| HypLiLoc  |     13.818     |     2.737     |    49.074     |     54.819      |         |
| APRBiCA   |     5.870      |     1.427     |    49.355     |     22.489      |         |


evaluated w. single 5060ti GPU
trained w. twp 3090 ti GPUs

------

# Reference

## PointLoc
```bibtex
@article{wang2021pointloc,
  title={Pointloc: Deep pose regressor for lidar point cloud localization},
  author={Wang, Wei and Wang, Bing and Zhao, Peijun and Chen, Changhao and Clark, Ronald and Yang, Bo and Markham, Andrew and Trigoni, Niki},
  journal={IEEE Sensors Journal},
  volume={22},
  number={1},
  pages={959--968},
  year={2021},
  publisher={IEEE}
}
```


## PosePN, PosePN++, PoseMinkLoc, PoseSOE
```bibtex
@article{yu2022lidar,
  title={LiDAR-based localization using universal encoding and memory-aware regression},
  author={Yu, Shangshu and Wang, Cheng and Wen, Chenglu and Cheng, Ming and Liu, Minghao and Zhang, Zhihong and Li, Xin},
  journal={Pattern Recognition},
  volume={128},
  pages={108685},
  year={2022},
  publisher={Elsevier}
}
```


## STCLoc
[Official Git Repo](https://github.com/PSYZ1234/STCLoc)
```bibtex
@article{yu2022stcloc,
  title={Stcloc: Deep lidar localization with spatio-temporal constraints},
  author={Yu, Shangshu and Wang, Cheng and Lin, Yitai and Wen, Chenglu and Cheng, Ming and Hu, Guosheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={24},
  number={1},
  pages={489--500},
  year={2022},
  publisher={IEEE}
}
```


## HypLiLoc
[Official Git Repo](https://github.com/sijieaaa/HypLiLoc?tab=readme-ov-file)
```bibtex
@inproceedings{wang2023hypliloc,
  title={HypLiLoc: Towards Effective LiDAR Pose Regression with Hyperbolic Fusion},
  author={Wang, Sijie and Kang, Qiyu and She, Rui and Wang, Wei and Zhao, Kai and Song, Yang and Tay, Wee Peng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5176--5185},
  year={2023}
}
```

## APR-BiCA
```bibtex
@article{dai2025apr,
  title={APR-BiCA: LiDAR-based absolute pose regression with bidirectional cross attention and gating unit},
  author={Dai, Jianlong and Wang, Hui and Zhao, Yuqian and Liu, Zhihua},
  journal={Neurocomputing},
  pages={132404},
  year={2025},
  publisher={Elsevier}
}
```
