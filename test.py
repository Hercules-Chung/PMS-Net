import numpy as np
import cv2
from keras.models import load_model
# import scipy.io as sio
from dehaze_patchMap_dehaze import dehaze_patchMap

base_path_hazyImg = 'image/'
base_path_result = 'patchMap/'
imgname = 'waterfall.tif'
save_dir = 'result/'
modelDir = 'PMS-Net.h5'

print("Process image: ", imgname)

image = cv2.imread(base_path_hazyImg + imgname)

if image.shape[0] != 480 or image.shape[1] != 640:
    print('resize image tp 640*480')
    image = cv2.resize(image, (640, 480))
hazy_input = np.reshape(image, (1, 480, 640, 3))
model = load_model(modelDir)
patchMap = model.predict(hazy_input, verbose=1)
patchMap = np.reshape(patchMap, (480, 640))


savename_result = save_dir + 'py_recover_new' + imgname.split()[0] + '.bmp'


recover_result, tx = dehaze_patchMap(image, 0.95, patchMap)

cv2.imshow('res', recover_result/255)
cv2.waitKey()
# print(np.rint(recover_result).astype('uint8'))
# cv2.imwrite(savename_result, np.rint(recover_result).astype('uint8'))
cv2.imwrite(savename_result, recover_result)
# print(recover_result.astype('float32')/255)
# print((recover_result.astype('float32')/255).dtype)
# print((recover_result/255).astype('float32'))

# sampleRst = cv2.imread('py_recover_waterfall.tif')
# diff = recover_result - sampleRst
# savename_result = save_dir + 'py_recover_'+ '_'+ imgname
# cv2.imwrite(savename_result, recover_result)
# print(diff.min(), diff.max())
# print(recover_result.astype('float32')/255)
# cv2.imshow('shit', recover_result.astype('float32')/255)
# cv2.waitKey()

# print(recover_result.dtype)
# a = 'float32'
# while a != '':
#     savename_result = save_dir + 'py_recover_' + a + '_'+ imgname + 'f'
#     if a == "rint":
#         neNp = np.rint(recover_result).astype('uint8')
#         cv2.imwrite(savename_result, neNp)
#         print(neNp)
#         print(neNp.dtype)
#     else:
#         try:
#             neNp = recover_result.astype(a)
#             cv2.imwrite(savename_result, neNp)
#             print(neNp)
#             print(neNp.dtype)
#         except Exception as e:
#             print(e)
#     a = input('Input format\n')
