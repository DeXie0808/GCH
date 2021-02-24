import h5py
import numpy as np
from PIL import Image
import cPickle as pickle
import scipy.io as sio


# F1 = h5py.File('/home/xrq/Dataset/IJCAI19/Flickr/ImgIJCAI.h5', 'r')
# db = F1['ImgDataBase'][:]
# tr = F1['ImgTrain'][:]
# qu = F1['ImgQuery'][:]
#
# path_db_ori = []
# path_tr_ori = []
# path_qu_ori = []
#
# for i in range(len(db)):
#     path_db_ori.append(db[i].replace('/home/libsource/Cross_Modal_Datasets/', '/home/xrq/Dataset/'))
#
# for i in range(len(tr)):
#     path_tr_ori.append(db[i].replace('/home/libsource/Cross_Modal_Datasets/', '/home/xrq/Dataset/'))
#
# for i in range(len(qu)):
#     path_qu_ori.append(qu[i].replace('/home/libsource/Cross_Modal_Datasets/', '/home/xrq/Dataset/'))
#
# # path_tr = path_tr_ori + path_db_ori[-5000:]
# # path_db = path_db_ori[0:18348-5000]
# # path_qu = path_qu_ori
#
# F2 = h5py.File('/home/xrq/Dataset/IJCAI19/Flickr/ImgIJCAI_new.h5', 'w')
# F2.create_dataset('ImgDataBase', data=path_db_ori)
# F2.create_dataset('ImgTrain', data=path_tr_ori)
# F2.create_dataset('ImgQuery', data=path_qu_ori)
# F2.close()

# print 'Done!'
#
# F2 = h5py.File('/home/xrq/Dataset/IJCAI19/Flickr/ImgIJCAI.h5', 'r')
# db_n = F2['ImgDataBase'][:]
# tr_n = F2['ImgTrain'][:]
# qu_n = F2['ImgQuery'][:]
# print "loaded!"

# f = open('/home/xrq/Dataset/IJCAI19/Flickr/FlickrDict.pkl', 'rb')
# dt = pickle.load(f)
# f.close()
# a = np.dot(dt, dt.transpose())/np.max(np.dot(dt, dt.transpose()))
#
# sio.savemat('/home/xrq/Dataset/IJCAI19/Flickr/FlickrDict.mat', {'a': a})
# a = [1,2,3]
# b = [4,5,6]
# c = []
#
# for i in range(len(a)):
#     tmp = []
#     tmp.append(a[i])
#     tmp.append(b[i])
#     c.append(tmp)
# print c