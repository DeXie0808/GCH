import h5py
import numpy as np
import cPickle as pickle

select_gpu = '3'
per_process_gpu_memory_fraction = 0.9

MODEL_DIR = '/home/xrq/Dataset/imagenet-vgg-f.mat'
phase = 'train'
#--------------------------load data-----------------------------#
DATA_DIR = '/home/xrq/Dataset/IJCAI19/Flickr/'
Imgpath  = '/home/xrq/Dataset/IJCAI19/Flickr/ImgIJCAI.h5'
Tagpath  = '/home/xrq/Dataset/IJCAI19/Flickr/TagIJCAI.h5'
Labpath  = '/home/xrq/Dataset/IJCAI19/Flickr/LabIJCAI.h5'
Semanticpath  = '/home/xrq/Dataset/IJCAI19/Flickr/SemanticIJCAI.h5'
meanpath = '/home/xrq/Dataset/IJCAI19/Flickr/Mean.h5'

dataset_dir = 'Flickr'
# Data = LoadData(DATA_DIR, dataset_dir)

F = h5py.File(meanpath, 'r')
meanpix = F['Mean'][:]
F.close()

checkpoint_dir = './checkpoint'
sample_dir = './sample_dir'
test_dir = './test_dir'

#---------------------Train = Retrieval--------------------------#

F1 = h5py.File(Imgpath, 'r')
Img_train_path = F1['ImgTrain'][:]
Img_db_path = F1['ImgDataBase'][:]
Img_query_path = F1['ImgQuery'][:]

Img_data =np.load('/home/xrq/Dataset/IJCAI19/Flickr/Img.npz')
Img_train = Img_data['Img_train']
Img_db = Img_data['Img_db']
Img_query = Img_data['Img_query']

F2 = h5py.File(Tagpath, 'r')
Txt_train = F2['TagTrain'][:]
# Txt_retrieval = F2['TagDataBase'][:]
Txt_query = F2['TagQuery'][:]

F3 = h5py.File(Labpath, 'r')
Lab_train = F3['LabTrain'][:]
# Lab_retrieval = F3['LabDataBase'][:]
Lab_query = F3['LabQuery'][:]

F4 = h5py.File(Semanticpath, 'r')
Smt_train = F4['SmtTrain'][:]
# Lab_retrieval = F3['LabDataBase'][:]
Smt_query = F4['SmtQuery'][:]

Img_train = Img_train[0:10000]
Txt_train = Txt_train[0:10000]
Lab_train = Lab_train[0:10000]
Smt_train = Smt_train[0:10000]

idx = ind = np.random.choice(len(Img_db), size=5000)
# Img_retrieval = F1['ImgDataBase'][:][idx]
Img_retrieval = Img_db[idx]
Txt_retrieval = F2['TagDataBase'][:][idx]
Lab_retrieval = F3['LabDataBase'][:][idx]
Smt_retrieval = F4['SmtDataBase'][:][idx]

train_L = Lab_train.astype(np.float32)
train_x = Img_train
train_y = Txt_train.astype(np.float32)

query_L = Lab_query.astype(np.float32)
query_x = Img_query
query_y = Txt_query.astype(np.float32)

retrieval_L = Lab_retrieval.astype(np.float32)
retrieval_x = Img_retrieval
retrieval_y = Txt_retrieval.astype(np.float32)

bit = 16
alpha = 1
gamma = 1
beta = 1
eta = 10
delta = 1

save_freq = 1
Epoch = 500


num_train = len(Txt_train)
numClass = Lab_train.shape[-1]
dimText = Txt_train.shape[-1]
dimLab = numClass


SEMANTIC_EMBED = 512
MAX_ITER = 100
batch_size = 256
num_proposal = 8
image_size = 224

DATABASE_SIZE = 18015
TRAINING_SIZE = 10000
QUERY_SIZE = 2000
VERIFICATION_SIZE = 1000
N = dimLab


Sim = (np.dot(Lab_train, Lab_train.transpose()) > 0).astype(int) * 0.999
# embedding_matrix = pickle.load(file('/home/xrq/Dataset/IJCAI19/Flickr/FlickrDict.pkl', 'rb'))  # 24*300
# semantics_all = np.dot(Smt_train, Smt_train.transpose())

lr_img = 0.0001 #0.0001
lr_txt = 0.01 #0.001
lr_lab = 0.01
lr_gph = 0.001
# learn_rate = 0.01
decay = 0.9
decay_steps = 1