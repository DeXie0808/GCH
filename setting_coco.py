import numpy as np
import cPickle as pickle
import scipy.io
import h5py
from load_data import LoadData
# from load_data import split_data


# environmental setting: setting the following parameters based on your experimental environment.
# select_gpu = '0,1'
select_gpu = '3'
per_process_gpu_memory_fraction = 0.9

# Initialize data loader
MODEL_DIR = '/home/xrq/Dataset/imagenet-vgg-f.mat' #discriminator_img  pretrain model
DATA_DIR  = '/home/xrq/Dataset/IJCAI19/COCO/Img_new.h5'
Imgpath  = '/home/xrq/Dataset/IJCAI19/COCO/Img.h5'
Tagpath  = '/home/xrq/Dataset/IJCAI19/COCO/Tag.h5'
Labpath  = '/home/xrq/Dataset/IJCAI19/COCO/Lab.h5'
meanpath = '/home/xrq/Dataset/IJCAI19/COCO/Mean.h5'

CoCo = LoadData(DATA_DIR)
dataset_dir = 'CoCo'
CoCo_Mean = h5py.File(meanpath, 'r')
meanpix = CoCo_Mean['Mean'][:].astype(np.float32)
CoCo_Mean.close()

Img = h5py.File(Imgpath, 'r')
train_x = Img['ImgTrain'][:] #  Img['ImgDataBase'][:] #
retrieval_x = Img['ImgDataBase'][:]
query_x = Img['ImgQuery'][0:5000]

Tag = h5py.File(Tagpath, 'r')
train_y = Tag['TagTrain'][:].astype(np.float32) # Tag['TagDataBase'][:].astype(np.float32) #
retrieval_y = Tag['TagDataBase'][:].astype(np.float32)
query_y = Tag['TagQuery'][0:5000].astype(np.float32)

Lab = h5py.File(Labpath, 'r')
train_L = Lab['LabTrain'][:].astype(np.float32) # Lab['LabDataBase'][:].astype(np.float32) #
retrieval_L = Lab['LabDataBase'][:].astype(np.float32)
query_L = Lab['LabQuery'][0:5000].astype(np.float32)

dimTxt = train_y.shape[1]
dimLab = train_L.shape[1]

phase = 'train'
checkpoint_dir = './checkpoint'
Savecode = './Savecode'
# dataset_dir = 'FLICKR'
# netStr = 'alex'

num_train = train_L.shape[0]
numClass = train_L.shape[1]
dimText = train_y.shape[1]

SEMANTIC_EMBED = 512
MAX_ITER = 100
batch_size = 128
# num_proposal = 8
image_size = 224

N = dimLab

Sim = (np.dot(train_L, train_L.transpose()) > 0).astype(int)*0.999
# embedding_matrix = pickle.load(file('/home/xrq/Dataset/IJCAI19/Flickr/FlickrDict.pkl', 'rb'))  # 24*300
# semantics_all = np.dot(train_L, embedding_matrix)

lr_img = 0.0001 #0.0001
lr_txt = 0.01 #0.001
lr_lab = 0.01
lr_gph = 0.001
learn_rate = 0.0001
decay = 0.5
decay_steps = 1