# COSE474-Final-Project Report
_**This is a research project conducted as the final project for COSE474 Deep Learning.**_
## Model Structure
<p align="center">
  <img src="https://github.com/jwj51720/COSE474-Final-Project/assets/104672441/bb3ef797-8766-431d-9940-990e857dfd04" width="500">
</p>

## Motivation
Many real-world problems consist of multiple classes, and there are areas like healthcare where accurate class assignment becomes critical. In general, multi-class classification problems are less accurate than binary classification problems because it becomes more difficult to accurately learn and distinguish the boundaries between each class. This phenomenon becomes more pronounced as the number of classes that need to be distinguished increases, so finding a way to address it is the motivation for this research. Consequently, this study propose a novel model structure aimed at both reducing the complexity of distinguishing classes and enhancing accuracy through a mechanism that compensates for incorrect classifications.

## Results
<p align="center">
  <img src="https://github.com/jwj51720/COSE474-Final-Project/assets/104672441/a2d92278-ec7e-4b17-9ea9-de65ee6d27b9" width="300" style="display:inline-block; margin: 0 auto;">
  <img src="https://github.com/jwj51720/COSE474-Final-Project/assets/104672441/0e472529-b4af-409a-9f24-896cbab07102" width="300" style="display:inline-block; margin: 0 auto;">
  <img src="https://github.com/jwj51720/COSE474-Final-Project/assets/104672441/6dce6155-1afa-4c25-a39d-5fca13cad2f1" width="300" style="display:inline-block; margin: 0 auto;">
</p>

## How To Run
```
$ python main.py
$ python inference.py
```
- Before training, please configure the experimental settings in the `experimental_setting.json` file.
- The training logs are saved using the TensorBoard library. Please check the storage path.
