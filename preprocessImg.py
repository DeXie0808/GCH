from loadImg import *
import h5py
import numpy as np

Imgpath  = '/home/xrq/Dataset/IJCAI19/Flickr/ImgIJCAI.h5'
F1 = h5py.File(Imgpath, 'r')
Img_train_path = F1['ImgTrain'][:]
Img_db_path = F1['ImgDataBase'][:]
Img_query_path = F1['ImgQuery'][:]

Img_train = loadimg(Img_train_path)
Img_db = loadimg(Img_db_path)
Img_query = loadimg(Img_query_path)

np.savez('/home/xrq/Dataset/IJCAI19/Flickr/Img.npz', Img_train=Img_train, Img_query=Img_query, Img_db=Img_db)
# np.savez('/home/xrq/Dataset/IJCAI19/Flickr/Img_train.mat', {'Img_train': Img_train})
# sio.savemat('/home/xrq/Dataset/IJCAI19/Flickr/Img_query.mat', {'Img_query': Img_query})
# sio.savemat('/home/xrq/Dataset/IJCAI19/Flickr/Img_db.mat', {'Img_db': Img_db})