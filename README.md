# SkySegmentation

## Dataset conversion

Dataset need to be in LabelMe format. 
Images and JSON are in one directory.
To convert the images to a supported format, run `create_dataset.py`.

This requires **different conda env** witr **LabelMe** installed.
VS Code seems to have problems running another conda env in the terminal, so make sure to run it from an **outside terminal**.

````
python create_dataset.py    path/to/labelme     \
                            path/to/save        \
                            path/to/labels.txt  
````


## Training

Train the model using `train.py`.

Available architectures are:
- pfn
- unet
- deeplab


To train run:
````
python train.py     --data_dir    path/to/dataset         \
                    --save_dir    path/to/save/model      \
                    --num_classes 5                       
````