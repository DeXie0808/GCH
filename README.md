# GCH
Graph Convolutional Network Hashing for Cross-Modal Retrieval, IJCAI2019


### Introduction
we propose a Graph Convolutional Hashing (GCH) approach, which learns modality-unified binary codes via an affinity graph. For more details, please refer to our
[paper](https://see.xidian.edu.cn/faculty/chdeng/Welcome%20to%20Cheng%20Deng's%20Homepage_files/Papers/Conference/IJCAI2019_Ruiqing.pdf).

<!-- ![alt text](http://cs.rochester.edu/u/zyang39/VG_ICCV19.jpg 
"Framework") -->
<p align="center">
  <img src=fig/framework.png width="75%"/>
</p>

### Citation

    @inproceedings{xu2019graph,
    title={Graph Convolutional Network Hashing for Cross-Modal Retrieval.},
    author={Xu, Ruiqing and Li, Chao and Yan, Junchi and Deng, Cheng and Liu, Xianglong},
    booktitle={Ijcai},
    pages={982--988},
    year={2019}
    }

### Prerequisites

* Python 2.7
* Tensorflow 1.2.0
* Others (numpy, scipy, h5py, etc.)

## Installation

1. Clone the repository

    ```
    git clone https://github.com/DeXie0808/GCH.git
    ```

2. Prepare the submodules and associated data

* RefCOCO & ReferItGame Dataset: place the data or the soft link of dataset folder under ``./ln_data/``. We follow dataset structure [DMS](https://github.com/BCV-Uniandes/DMS). To accomplish this, the ``download_dataset.sh`` [bash script](https://github.com/BCV-Uniandes/DMS/blob/master/download_data.sh) from DMS can be used.
    ```bash
    bash ln_data/download_data.sh --path ./ln_data
    ```

<!-- * Flickr30K Entities Dataset: place the data or the soft link of dataset folder under ``./ln_data/``. The formated Flickr data is availble at [[Gdrive]](https://drive.google.com/open?id=1A1iWUWgRg7wV5qwOP_QVujOO4B8U-UYB), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Eqgejwkq-hZIjCkhrgWbdIkB_yi3K4uqQyRCwf9CSe_zpQ?e=dtu8qF).
    ```
    cd ln_data
    tar xf Flickr30k.tar
    cd ..
    ``` -->
* Flickr30K Entities Dataset: please download the images for the dataset on the website for the [Flickr30K Entities Dataset](http://bryanplummer.com/Flickr30kEntities/) and the original [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/). Images should be placed under ``./ln_data/Flickr30k/flickr30k_images``.


* Data index: download the generated index files and place them as the ``./data`` folder. Availble at [[Gdrive]](https://drive.google.com/open?id=1cZI562MABLtAzM6YU4WmKPFFguuVr0lZ), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Epw5WQ_mJ-tOlAbK5LxsnrsBElWwvNdU7aus0UIzWtwgKQ?e=XHQm7F).
    ```
    rm -r data
    tar xf data.tar
    ```

* Model weights: download the pretrained model of [Yolov3](https://pjreddie.com/media/files/yolov3.weights) and place the file in ``./saved_models``. 
    ```
    sh saved_models/yolov3_weights.sh
    ```
More pretrained models are availble in the performance table [[Gdrive]](https://drive.google.com/open?id=1-DXvhEbWQtVWAUT_-G19zlz-0Ekcj5d7), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/ErrXDnw1igFGghwbH5daoKwBX4vtE_erXbOo1JGnraCE4Q?e=tQUCk7) and should also be placed in ``./saved_models``.


### Training
3. Train the model, run the code under main folder. 
Using flag ``--lstm`` to access lstm encoder, Bert is used as the default. 
Using flag ``--light`` to access the light model.

    ```
    python train_yolo.py --data_root ./ln_data/ --dataset referit \
      --gpu gpu_id --batch_size 32 --resume saved_models/lstm_referit_model.pth.tar \
      --lr 1e-4 --nb_epoch 100 --lstm
    ```

4. Evaluate the model, run the code under main folder. 
Using flag ``--test`` to access test mode.

    ```
    python train_yolo.py --data_root ./ln_data/ --dataset referit \
      --gpu gpu_id --resume saved_models/lstm_referit_model.pth.tar \
      --lstm --test
    ```

5. Visulizations. Flag ``--save_plot`` will save visulizations.
