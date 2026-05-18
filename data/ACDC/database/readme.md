# ACDC data
It is in this folder where the data should be. As this data is quite large and I am not the creator of the data i havn't included it in this repo.
If you want the data i would refer to the [ACDC dataset website](https://www.creatis.insa-lyon.fr/Challenge/acdc/) and the data can be downloaded there. In the data given by the website there should be two folders that should be placed in this one. It is "testing" and "training". After they are placed here please run the preprocessing.py script. This should generate two folders, that being training_npy and testing_npy. These folder contain the resampled mri pluss they are less compressed so that runtime is faster. 

Running preprocessing.py:
1. Go to `if __name__ == "__main__"` and choose whether to resample the training data, test data, or both.
2. You can also configure the seed that you want for generating the folds inside of this script
3. Run the script.

There might also be two additional folders training_cross_val and testing_cross_val (or whatever you decide to call them) if you later run segsave.py in the root of the project. These folders contain predicted masks by the segmentation models. And can then be used for the classification models.