# AI_AFM
Repository for implementing AI in Atomic Force Microscopy (AFM)


## Sample plots
<p align="center">
    <img src="/images/0_set1_192.png" width="200">
    <img src="/images/1_set1_154.png" width="200">
    <img src="/images/2_set1_174.png" width="200">
    <img src="/images/3_set1_63.png" width="200">
</p>
The figures above shows the sample plots for (a) no rupture, (b) single rupture, (c) double rupture and (d) multiple rupture.


## Classification Labels
<center>

| Labels | 3-class | 2-class (single vs rest) | 2-class (no vs rest) |
|:-------:|:-------:|:-------:|:-------:|
| 0 | No rupture and multiple rupture | Rest | No rupture |
| 1 | Single rupture | Single rupture | Rest |
| 2 | Double rupture rupture | - | - |

</center>

## Algorithm Overview
<p align="center">
    <img src="/images/Few-shot Architecture-top k.jpg" width="800">
</p>
The figure above shows our few-shot framework. Training, shown in the top block, is done by inputting three samples from the train set into a triplet loss architecture. Testing, shown in the bottom block, is done by inputting train-test sample pairs into the model to compare embedding distances.

## Results
<p align="center">
    <img src="/images/metrics_comp4.JPG" width="800">
</p>
Accuracy, class accuracies, precision, recall, and F-1 score for our few-shot approach, SVM, RF, and KNN are shown above for the 3-class case.

<br />
<br />

<p align="center">
    <img src="/images/metrics_comp_2class.JPG" width="600">
</p>
Accuracy and class accuracies are shown above for our few-shot approach on 2-class scenarios.

## Loss Functions
Assuming Y = binary class label where similar class label = 1, dissimilar class label = 0, m = margin (default = 1.0) and let distance $D_{1,2} = (embedding_1 - embedding_2)^2$, 

### Simplified Contrastive Loss (ContrastiveLoss_ori):

$$L = \left( YD \right) + \left( 1-Y \right) \left[ max \left( 0, m-D \right)\right]$$


### Modified Contrastive Loss (ContrastiveLoss):

$$L = 0.5 \left[ \left( YD \right) + \left( 1-Y \right) max \left(0, m-\sqrt{D} \right)^2 \right]$$


### Triplet Loss (TripletLoss)

$$L =  max \left( 0, D_{(+ve, anchor)} - D_{(-ve, anchor)} + m \right)$$

## Install
Clone repo and install requirements.txt in a Python>=3.9.12 enviornment.
~~~
git clone https://github.com/JoshuaRWaite/AI_AFM  # clone
cd AI_AFM
pip install -r requirements.txt  # install
~~~

## Running few-shot experiments
Example run for 3 class (No/Multipe Rupture, Single Rupture, Double Rupture) case: 
~~~
python main.py -d 3class_matching -m convo1D2 -mod 10 -ep 300 -bs 32 -opt sgd --lr 0.01 -sche_step 1 -sche_gamma 1 -mt triplet -s 0 -exp 4
~~~

Example run for 2 class (Single Rupture vs Rest) case: 
~~~
python main.py -d 2class_s_vs_r -m convo1DS2 -mod 10 -ep 300 -bs 32 -opt sgd --lr 0.01 -sche_step 1 -sche_gamma 1 -mt triplet -g 0 -s 0 -up 0 -exp 5
~~~

Example run for 2 class (No Rupture vs Rest) case: 
~~~
python main.py -d 2class_n_vs_r -m convo1DDrp2 -mod 10 -ep 300 -bs 16 -opt sgd --lr 0.01 -sche_step 1 -sche_gamma 1 -mt triplet -g 0 -s 0 -up 0 -exp 6
~~~

## Running few-shot evaluation with pretrained weights
Example eval for 3 class (No/Multipe Rupture, Single Rupture, Double Rupture) case: 
~~~
python eval.py -d 3class_matching -m convo1D2 -mod 10 -ep 300 -bs 32 -opt sgd --lr 0.01 -sche_step 1 -sche_gamma 1 -mt triplet -s 0 -exp 1 -tar 28-06-2022_220804_best
~~~

Example eval for 2 class (Single Rupture vs Rest) case: 
~~~
python eval.py -d 2class_s_vs_r -m convo1DS2 -mod 10 -ep 300 -bs 32 -opt sgd --lr 0.01 -sche_step 1 -sche_gamma 1 -mt triplet -g 0 -s 0 -up 0 -exp 2 -tar 28-06-2022_163140_best
~~~

Example eval for 2 class (No Rupture vs Rest) case: 
~~~
python eval.py -d 2class_n_vs_r -m convo1DDrp2 -mod 10 -ep 300 -bs 16 -opt sgd --lr 0.01 -sche_step 1 -sche_gamma 1 -mt triplet -g 0 -s 0 -up 0 -exp 3 -tar 28-06-2022_163633_best
~~~

## List of Optimizer (-opt argument)
- SGD (sgd)
- ADAM (adam)

## List of Loss Functions (-mt argument)
- Contrastive (siam)
- Triplet (triplet)

## List of Models (-m argument)
- Linear (toy, toyL, toyS, toyS2, toyS3, toyXS, cerb, cerbL, cerbXL)
- Convolutional (convo1D, convo1DS, convo1DDrp, convo1DDrp2, convo1DS2, convo1D2, convo1D3)

## List of Datasets (-d argument)
- No/multiple rupture, single rupture, and double rupture (3class_matching)
- Single rupture vs rest (2class_s_vs_r)
- No rupture vs rest (2class_n_vs_r)

## Running shallow experiments
Example run for shallow methods with SVM and varying percentage of noisy data for the 3 class case: 
~~~
python re_shallow.py -mode noise -algo SVM -d 3class_matching
~~~
-mode has noise and trainsz
<br />
-algo has KNN, RF, and SVM

## Acknowledgements
We would like to acknowledge Prof. Peter Hoffmann (Department of Physics, Wayne State University) and Prof. Rafael Fridman (Department of Pathology, Wayne State University) for their help in culturing cells with overexpressed receptor and performing AFM force-distance measurements. This work was supported by Iowa State College of Engineering Exploratory Research Program (A.S., S.S.).


## Citation
Please cite our paper in your publications if it helps your research:

	@article{waite_tan_saha_sarkar_sarkar_2023,
	  title={Few-shot deep learning for AFM Force curve characterization of single-molecule interactions},
	  author={Waite, Joshua R. and Tan, Sin Yong and Saha, Homagni and Sarkar, Soumik and Sarkar, Anwesha},
	  journal={Patterns},
	  year={2023}
	}

## Paper Links
[Few-shot deep learning for AFM force curve characterization of single-molecule interactions](https://doi.org/10.1016/j.patter.2022.100672)




## Contributors
- [Sin Yong Tan](https://github.com/tsyong98)
- [Joshua Waite](https://github.com/JoshuaRWaite)
