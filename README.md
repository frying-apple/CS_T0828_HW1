# CS_T0828_HW1

## Setup

1. Using: Windows, Anaconda, PyCharm
2. Create PyCharm project via pull from https://github.com/frying-apple/CS_T0828_HW1.git and set \<project folder\>
3. Download data from https://www.kaggle.com/c/cs-t0828-2020-hw1/data
4. Extract like: \<project folder\>\\cs-t0828-2020-hw1\{training_labels.csv, training_data, testing_data}
5. Set Conda environment in PyCharm (see next section)

## Requirements
Make a conda environment in a path without spaces.  I use: C:\anaconda3\envs\CS_T0828_HW1

pip install tensorflow==2.3.1

Also, install other packages listed in preprocessing.py

I am using an old version of CUDNN, etc. so will get "Could not load dynamic library"; therefore, do this extra step:

Copy the following dll from somewhere into: C:\anaconda3\envs\CS_T0828_HW1\Library\bin\ 

cudart64_101.dll, 
cublas64_10.dll, 
cufft64_10.dll, 
curand64_10.dll, 
cusolver64_10.dll, 
cusparse64_10.dll, 
cudnn64_7.dll

Then, the beginning section of preprocessing.py should return "Num GPUs Available:  1" in the console.

## Run
Run preprocessing.py

This will preprocess, train, and save the model.

Run test.py

This will preprocess (again), load the model, and test it.  See 'test.csv' for result.