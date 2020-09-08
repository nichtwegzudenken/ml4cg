# FUNIT2FUNIT: Assessing the effect of the G2G Architecture on Disentanglement

### G2G - Disentanglement by Cross-Training

![GitHub Logo](/images/funit_cross_forward_pass.png)

This is a repository for the class "Machine Learning for Computer Graphics" at Tel-Aviv University and uses Liu et al.'s FUNIT model as a baseline for our cross-training approach [[1]](#1).
The baseline code for the FUNIT project has been taken from: https://github.com/NVlabs/FUNIT.

## Installation

* Clone this repo using ```https://github.com/nichtwegzudenken/ml4cg.git```
* Follow Installation Instructions from the FUNIT page [FUNIT](https://github.com/NVlabs/FUNIT)
* Install [pytorch](https://pytorch.org/) by following instructions on their website

## Dataset

### Animal Face Dataset

We use the Animal Face Dataset which was also used in the FUNIT paper [[1]](#1) and can be found under [FUNIT](https://github.com/NVlabs/FUNIT) as well.

## Training

Once the dataset is prepared and lies in /dataset/animals the training can be started with the following command:

```
python train.py --config configs/funit_animals.yaml --multigpus
```

The output images are then located in the folder outputs/funit_animals/images and the logs in /logs/funit_animals. Also checkpoints and intermediate results can be found in outputs/funit_animals.
To adjust training parameters the configs/funit_animals.yaml file can be adjusted.

## Testing Forward Pass through FUNIT2FUNIT

To obtain the pretrained model go to the [FUNIT](https://github.com/NVlabs/FUNIT) page and download the model.
To run a forward pass through the G2G architecture run:

```
python test_1_shot_g2g.py --config configs/funit_animals.yaml --ckpt pretrained/animal149_gen.pt --input images/input_content.jpg --class_image_folder images/n02138411 --output images/g2g_1_shot.jpg
```
The command above will take the images x1 and x2 as inputs and output four images m1, m2, r1 and r2, which correspond to the two mixed images and the two reconstructed images.

![GitHub Logo](/images/funit2funit.png)

The two mixed images are:

![GitHub Logo](/images/m1.jpg)

![GitHub Logo](/images/m2.jpg)

With the reconstructed images resulting in:

![GitHub Logo](/images/r1.png)

![GitHub Logo](/images/r2.png)

## References
<a id="1">[1]</a> 
Liu, Ming-Yu, et al. (2019). 
Few-shot unsupervised image-to-image translation.
Proceedings of the IEEE International Conference on Computer Vision. 2019.
