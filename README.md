# ftcnn_reid
Currently, only feature extraction part is avaibale. 

requirement pycaffe

## fine-tuned model on PETA dataset
* deploy file of AlexNet 
* AlexNet fine-tuned on PETA except VIPeR
* AlexNet fine-tuned on PETA except CUHK
* AlexNet fine-tuned on PETA except GRID
* AlexNet fine-tuned on PETA except PRID
* AlexNet fine-tuned on full PETA dataset

## Usage
To extract features, download database, CNN models and put to adequate place. 
Modify the directly file of extract_features.py. 
And then run the following command. 

```
python extract_features.py
```
