# ftcnn_reid
Currently, only feature extraction part is avaibale. 

requirement pycaffe

## fine-tuned model on PETA dataset
* [deploy file of Alexnet](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/Alexnet_deploy.prototxt) 
* [AlexNet fine-tuned on PETA except VIPeR](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/Alexnet_PETA_except_VIPeR_comb_r7+multi_alpha0.5_train_iter_50000.caffemodel) 
* [AlexNet fine-tuned on PETA except CUHK](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/Alexnet_PETA_except_CUHK_comb_r7+multi_alpha0.5_train_iter_50000.caffemodel) 
* [AlexNet fine-tuned on PETA except GRID](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/Alexnet_PETA_except_GRID_comb_r7+multi_alpha0.5_train_iter_50000.caffemodel) 
* [AlexNet fine-tuned on PETA except PRID](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/Alexnet_PETA_except_PRID_comb_r7+multi_alpha0.5_train_iter_50000.caffemodel) 
* [AlexNet fine-tuned on full PETA dataset](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/Alexnet_PETA_all_comb_r7+multi_alpha0.5_train_iter_50000.caffemodel) 

## Usage
To extract features, download database, CNN models and put to adequate place. 
Modify the directly file of extract_features.py. 
And then run the following command. 

```
python extract_features.py
```

## Extracted features.
* [Extracted features on Market-1501](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/features_FTCNN_Market.zip)
* [Extracted features on Duke](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/features_FTCNN_Duke.zip)


## Reference
Please cite the following [ICPR2016 paper](http://www.i.kyushu-u.ac.jp/~matsukawa/ReID_files/icpr2016.pdf).

```
@inproceedings{Matsukawa16b,
  author={Matsukawa, Tetsu and Suzuki, Einoshin},
  title={Person Re-Identification Using {CNN} Features Learned from Combination of Attributes},
  booktitle={{International Conference on Pattern Recognition (ICPR)}},
  pages={2429--2343}, 
  year={2016},
}

```
