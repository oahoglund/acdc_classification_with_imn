# Segmentation-Based Cardiac MRI Classification with an Interpretable Mesomorphic Network
This repo was written for my bachelors thesis in mathematical modelling and datascience in Oslo Metropolitan University. The thesis is in the Norwegian Research Information Repository: [link](https://hdl.handle.net/11250/5553654).
The thesis is derived from the EU-funded SEARCH project (https://ihi-search.eu), which aims to develop realistic and privacy-preserving synthetic data for safe and effective AI applications. 
One part of this involves the use of high-quality biomedical imaging open-source datasets, including cardiac magnetic resonance imaging (CMRI).
Another is the goal of developing explainable AI/ML tools that provide informed decisions to medical practitioners.

### Abstract of the thesis
Cardiovascular disease (CVD) is a leading cause of death worldwide and its prevalence is projected to grow considerably in the coming decades. Interpreting cardiac magnetic resonance imaging (CMRI) is labor intensive, and while machine learning has shown promise in automating this process, most models operate as black boxes, offering little insight into their predictions. This thesis investigates whether an inherently locally explainable model, an interpretable mesomorphic neural network (IMN), can match the predictive performance of black box approaches for cardiac pathology classification from CMRI. The proposed pipeline first segments cardiac structures using a U-Net, a convolutional encoder-decoder architecture, achieving macro-averaged Dice scores of 0.891 on the test set, with the left ventricle performing best and the myocardium worst. Clinical features are then extracted from the segmentation masks and used for classification. On the test set, the best performing black box model achieved an accuracy of 0.820 and an F1-score of 0.816, while the IMN achieved 0.760 and 0.760 respectively, with the two models performing more closely using cross-validation. The IMN feature importance patterns were broadly consistent with the known clinical characteristics of the pathologies, suggesting the explanations are clinically meaningful. These results indicate that interpretability need not come at a performance cost in the CMRI context.

### What is the purpose of this repo
To show my work and what I did with the data. Hopefully this repo is also enough for anyone else to reproduce my work. I try to strive for reproducible science.

### How to run:
1. Download the python libraries within requirements.txt and check with torch_check that a device (GPU) is being registered. This project is written with the assumption that you have a GPU. If you still want to run it without one, change the number of workers in the trainers in main.py.
2. Download data from the ACDC website.
3. Follow instructions inside data/ACDC/database and run preprocessing.py.
4. In main.py, run `cross_segmenter2()` (already set up in `__main__`) with your desired hyperparameters. This trains the segmentation model with cross-validation and automatically saves the checkpoint info to results/cross_val/segmentation/unet/info.json.
5. Run segsave.py. This will save the predicted segmentation masks for the held-out fold of each cross-validation split.
6. In main.py, run `master_run()` with your desired hyperparameters. This trains all classification models.
7. In main.py, run `test_master_run()`. This generates results for the test set.
8. Run xai.py to generate feature importance plots for the IMN model.

Additional notes:
- If you want the figures that appear in the first part of the methodology section, run the acdc_data_exploration notebook from top to bottom.
- If you want to regenerate confusion matrix PDFs from saved JSON files, run plot_conf_matrices.py and fill in the file paths needed in it.
- I have not actually tested this project on any other system than my own so expect bugs if you try to run this on your own. Make sure to change the torch version in the requirements if you use another version, especially nvidia

### Project structure
```
main.py                         — training and evaluation entry point (cross-validation, test set)
segsave.py                      — saves per-fold predicted segmentation masks for the training set
xai.py                          — feature importance analysis for the IMN model
plot_conf_matrices.py           — regenerates confusion matrix PDFs from saved JSON files
acdc_data_exploration.ipynb     — data exploration and methodology figures
torch_check.py                  — checks that a GPU is visible to PyTorch

src/
  acdc_segmentation2.py         — segmentation model (U-Net via segmentation-models-pytorch)
  classificationbaseline.py     — baseline CNN classification model
  clas_from_mask.py             — CNN classification model using segmentation masks as input
  clas_medical_metrics_mlp.py   — MLP classification model using clinical metrics
  clas_medical_metrics_imn.py   — interpretable mesomorphic network (IMN) classification model
  datasets/
    acdc_dataset_patients.py    — patient-level dataset (3D volumes, used for classification)
    acdc_dataset_slices.py      — slice-level dataset (2D slices, used for segmentation training)
    acdc_dataset_seg_pred.py    — dataset for generating segmentation predictions
  datamodules/
    acdc_dm.py                  — Lightning DataModule for classification and segmentation
    acdc_dm_pred.py             — Lightning DataModule for segmentation inference (segsave)
  utils/
    transform.py                — augmentation pipelines
    clinical_metrics.py         — clinical feature extraction (LVEF, RVEF, volumes, mass)
    logger.py                   — local image logger, run directory management, checkpoint lookup

data/
  ACDC/database/
    preprocessing.py            — converts raw ACDC NIfTI files to .npy format
    training_npy/               — preprocessed training patients (generated by preprocessing.py)
    testing_npy/                — preprocessed test patients
    training_cross_val/         — per-fold predicted masks for training patients (generated by segsave.py)
    testing_cross_val/          — ensemble predicted masks for test patients (generated by main.py)

results/
  cross_val/                    — cross-validation metrics, confusion matrices, and checkpoint info
  test_set/                     — test set metrics, confusion matrices, and visualizations

figures/                        — dataset exploration figures (generated by acdc_data_exploration.ipynb)
```

### Disclosure
Large language models (Claude and ChatGPT) were used as a coding aid in this repository, including for autocompletion, bug fixing, and prototyping. And it was also used in order to help fix language in this markdown file

### Citation
Plain text:
Höglund, Oscar Albert (2026). Segmenteringsbasert hjerte-MRI-klassifisering med et tolkbart mesomorft nettverk. https://hdl.handle.net/11250/5553654

Bibtex:
@mastersthesis{019fdc00b387-896e330c-010f-4502-ac19-aba2545ca388,
  author = {Höglund, Oscar Albert},
  month = {may},
  note = {nva type: DegreeBachelor},
  nva_api = {https://api.nva.unit.no/publication/019fdc00b387-896e330c-010f-4502-ac19-aba2545ca388},
  school = {OsloMet - storbyuniversitetet},
  title = {Segmenteringsbasert hjerte-MRI-klassifisering med et tolkbart mesomorft nettverk},
  url = {https://api.nva.unit.no/publication/019fdc00b387-896e330c-010f-4502-ac19-aba2545ca388},
  year = {2026}
}
### License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Note: this project depends on [albumentationsx](https://pypi.org/project/albumentationsx/), which is licensed under AGPL-3.0. If you intend to use this code in a commercial or proprietary setting, you will need a commercial license for albumentationsx.
