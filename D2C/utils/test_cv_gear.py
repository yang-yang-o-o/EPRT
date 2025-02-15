import cv2
import ctypes
from ctypes import *
import numpy as np
import mouse_cut

testso = CDLL("./libtest_localization.so")
frame = cv2.imread("./RGB_view_41.bmp")

# ROI =  mouse_cut.image_cut().cut(frame)

# cv2.imwrite('1.png',ROI)
# cv2.imshow("s",frame)
            # b"0.008,-0.04,0.267"
# cv2.waitKey(0)

# frame = np.asarray(frame, dtype=np.uint8) # 这句不要也可以


frame = frame.ctypes.data_as(ctypes.c_char_p) # 加这一句的目的是将frame的内存转换为连续存储，这样在通过unsigned char* 传递时才能传递完整的图片。
# frame = ROI.ctypes.data_as(ctypes.c_char_p) # 加这一句的目的是将frame的内存转换为连续存储，这样在通过unsigned char* 传递时才能传递完整的图片。
testso.d2co(
            ##### model
            b"3D_models/Gear.stl",
            ##### cfg
            b"test_images/test1_1_camera_calib.yml",
            ##### img
            frame,
            ##### R
            b"0.26758555,-0.53488473,0.05784623", # 最接近
            # b"-1.9,1.3,0.48",
            ##### t
            # b"0.008,-0.04,0.267"
            b"0.08163072436711435,-0.02079111854001027,0.35"
            # b"-0.1,-0.1,0.267"
            )