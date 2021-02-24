import numpy as np
from PIL import Image
from tqdm import tqdm

def loadimg(pathList):
    crop_size = 224
    ImgSelect = np.ndarray([len(pathList), crop_size, crop_size, 3])
    count = 0
    for path in tqdm(pathList):
        img = Image.open(path)
        xsize, ysize = img.size
        # **********************************************************************************************************
        # Here, we fist resize the original iamge into M*224 or 224*M, M>224, then cut out the part of M-224 surround.
        # seldim = min(xsize, ysize)
        # rate = 224.0 / seldim
        # img = img.resize((int(xsize * rate), int(ysize * rate)))
        # nxsize, nysize = img.size
        # box = (nxsize / 2.0 - 112, nysize / 2.0 - 112, nxsize / 2.0 + 112, nysize / 2.0 + 112)
        # img = img.crop(box)
        # img = img.convert("RGB")
        # img = img.resize((224, 224))
        # img = array(img)
        # if img.shape[2] != 3:
        # 	print 'This image is not a rgb picture: {0}'.format(pathList[idx])
        # 	print 'The shape of this image is {0}'.format(img.shape)
        # 	ImgSelect[count, :, :, :] = img[:, :, 0:3]
        # 	count += 1
        # else:
        # 	ImgSelect[count, :, :, :] = img
        # 	count += 1
        # **********************************************************************************************************
        nulArray = np.zeros([224, 224, 3])
        seldim = max(xsize, ysize)
        rate = 224.0 / seldim
        nxsize = int(xsize * rate)
        nysize = int(ysize * rate)
        if nxsize % 2 != 0:
            nxsize = int(xsize * rate) + 1
        if nysize % 2 != 0:
            nysize = int(ysize * rate) + 1
        img = img.resize((nxsize, nysize))
        nxsize, nysize = img.size
        img = img.convert("RGB")
        img = np.array(img)
        nulArray[112 - nysize / 2:112 + nysize / 2, 112 - nxsize / 2:112 + nxsize / 2, :] = img
        if nulArray.shape[2] != 3:
            print 'This image is not a rgb picture: {0}'.format(path)
            print 'The shape of this image is {0}'.format(nulArray.shape)
            ImgSelect[count, :, :, :] = nulArray[:, :, 0:3]
            count += 1
        else:
            ImgSelect[count, :, :, :] = nulArray
            count += 1
    return ImgSelect