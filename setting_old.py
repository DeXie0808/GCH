import numpy as np
import cPickle as pickle
import scipy.io
import h5py
from load_data import loading_data
from load_data import split_data


# environmental setting: setting the following parameters based on your experimental environment.
# select_gpu = '0,1'
select_gpu = '1'
per_process_gpu_memory_fraction = 0.9

# Initialize data loader
MODEL_DIR = '/data/xrq/imagenet-vgg-f.mat' #discriminator_img  pretrain model
# DATA_DIR  = '/home/xrq/Dataset/IJCAI19/COCO/Img.h5'
DATA_DIR  = '/data/xrq/FLICKR-25K.mat'
meanpath = '/data/xrq//Dataset/IJCAI19/Flickr/Mean.h5'

F = h5py.File(meanpath, 'r')
meanpix = F['Mean'][:]
F.close()

phase = 'train'
checkpoint_dir = './checkpoint'
Savecode = './Savecode'
dataset_dir = 'FLICKR'
netStr = 'alex'


SEMANTIC_EMBED = 512
MAX_ITER = 100
batch_size = 128
num_proposal = 8
image_size = 224


images, tags, labels = loading_data(DATA_DIR)
dimTxt = tags.shape[1]
dimLab = labels.shape[1]

DATABASE_SIZE = 18015
TRAINING_SIZE = 10000
QUERY_SIZE = 2000
VERIFICATION_SIZE = 1000
N = dimLab

X, Y, L = split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE)


train_L = L['train'].astype(np.float32)
train_x = X['train'].astype(np.float32)
train_y = Y['train'].astype(np.float32)

query_L = L['query'].astype(np.float32)
query_x = X['query'].astype(np.float32)
query_y = Y['query'].astype(np.float32)

# idx = np.random.permutation(5000)
# retrieval_L = L['retrieval'][idx,:].astype(np.float32)
# retrieval_x = X['retrieval'][idx,:].astype(np.float32)
# retrieval_y = Y['retrieval'][idx,:].astype(np.float32)


retrieval_L = L['retrieval'].astype(np.float32)
retrieval_x = X['retrieval'].astype(np.float32)
retrieval_y = Y['retrieval'].astype(np.float32)


# # Save dataset
# F1 = h5py.File('Tag.h5','w')
# F1.create_dataset('TagTrain', data = Y['train'])
# F1.create_dataset('TagQuery', data = Y['query'])
# F1.create_dataset('TagDataBase', data = Y['retrieval'])
# F1.close()
#
# F2 = h5py.File('Lab.h5','w')
# F2.create_dataset('LabTrain', data = L['train'])
# F2.create_dataset('LabQuery', data = L['query'])
# F2.create_dataset('LabDataBase', data = L['retrieval'])
# F2.close()
#
# F3 = h5py.File('Img.h5','w')
# F3.create_dataset('ImgTrain', data = X['train'])
# F3.create_dataset('ImgQuery', data = X['query'])
# F3.create_dataset('ImgDataBase', data = X['retrieval'])
# F3.close()

data = scipy.io.loadmat(MODEL_DIR)
imgMean = data['normalization'][0][0][0].astype(np.float32)
bit = 16
alpha = 1
gamma = 1
beta = 1
eta = 10
delta = 1

save_freq = 1
Epoch = 500


num_train = train_x.shape[0]
numClass = train_L.shape[1]
dimText = train_y.shape[1]


Sim = (np.dot(train_L, train_L.transpose()) > 0).astype(int)*0.999
# embedding_matrix = pickle.load(file('/home/xrq/Dataset/IJCAI19/Flickr/FlickrDict.pkl', 'rb'))  # 24*300
# semantics_all = np.dot(train_L, embedding_matrix)

lr_img = 0.0001 #0.0001
lr_txt = 0.01 #0.001
lr_lab = 0.01
lr_gph = 0.001
# learn_rate = 0.0001
decay = 0.5
decay_steps = 1