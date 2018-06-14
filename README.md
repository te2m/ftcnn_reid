# ftcnn_reid
Currently, only feature extraction part is avaibale. 

requirement pycaffe

## fine-tuned model on PETA dataset
* [deploy file of Alexnet](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/Alexnet_deploy.prototxt) 
* [AlexNet fine-tuned on PETA except VIPeR] 
(http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_file/Alexnet_PETA_except_VIPeR_comb_r7+multi_alpha0.5_train_iter_50000.caffemodel) 
* [AlexNet fine-tuned on PETA except CUHK]
* [AlexNet fine-tuned on PETA except GRID]
* [AlexNet fine-tuned on PETA except PRID]
* [AlexNet fine-tuned on full PETA dataset]

## Usage
To extract features, download database, CNN models and put to adequate place. 
Modify the directly file of extract_features.py. 
And then run the following command. 

```
python extract_features.py
```
