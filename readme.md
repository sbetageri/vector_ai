# Fashion MNIST

1. Setup
2. Downloading dataset into required format
3. Training and Saving Model
4. Testing Model
5. Inference on Model

## 1. Setup

1. Build docker image for current project <br>```docker build -t vector -f ./Dockerfile .```
2. Run docker image <br> ```docker run -it -v <path_to_code_dir>:/root/code --gpus all vector```

## 2. Downloading dataset
Downloads, splits and saves the fashion mnist data in the following format

There will be 2 folders created, **train** and **test**

The val_perc flag during training splits the training data into train/val. Test folder is used as a holdout.

```
root:
-class_1:
    -img_a
    -img_b
    -img_c
-class_2:
    -img_a
    -img_b
    -img_c
...
```

python3 download_fmnist.py --root <path_to_dir>

## 3. Training and saving model

Trains a simple ResNet 18 model after replacing the final fully connected layer. 

Training folder has to be in the ImageFolder(pytorch's ImageFolder) format.

Trained model is saved in a directory with a .pt extension

python3 train_model.py --train <path_to_train_dir> -v <val_split_percentage> -e <training_epochs> --out <dir_to_save_model>

## 4. Testing Model on Holdout Set

Tests a given model on a holdout set. 

Holdout set has to be in ImageFolder format

python3 test_model.py --model <path_to_model> --test <path_to_holdout_folder>

## 5. Inference

Infer the fashion item of the given image.

Ideally, the labels would be loaded up along with the model too. But I've hardcoded it here. 

python3 infer_model.py -m <path_to_model> -i <path_to_image>
