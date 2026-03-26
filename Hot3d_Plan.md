# Plan to integrate HOT3d Dataset
The conda environment hamer_dino_train is already set to work with HOT3D. The conda env hamer is also compatible.

The dataset loading code for hot 3d is located in third-party/hot3d, as well as most of the api and a tutorial HOT3D_Tutorial.ipynb which should be used to create the dataset loader objects. The dataset itself is still located at /home/yifanli/hot3d/hot3d/dataset.

I need the dataset loading code to allow me to run eval.py with the HOT3d dataset. The dataset is split across participants in the following format:
  dataset/                                                                                                                                                                                                                                              
  ├── P0001_10a27bf7/        ← original data
  ├── P0002_016222d1/        ← original data                                                                                                                                                                                                            
  ├── ...         
  ├── train/                                                                                                                                                                                                                                            
  │   ├── P0001_10a27bf7 → ../P0001_10a27bf7   (symlink)
  │   ├── P0002_016222d1 → ../P0002_016222d1   (symlink)                                                                                                                                                                                                
  │   └── ...     
  └── test/                                                                                                                                                                                                                                             
      ├── P0009_02511c2f → ../P0009_02511c2f   (symlink)
      └── ...                                              