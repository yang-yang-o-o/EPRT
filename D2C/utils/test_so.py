import ctypes
# from ctypes import *
# import cv2.aruco as aruco

testso = ctypes.cdll.LoadLibrary("./libtest_localization.so")

testso.d2co(b"3D_models/AX-01b_bearing_box.stl",
            b"test_images/test_camera_calib.yml",
            b"test_images/2.png",
            b"0,0,1",
            b"0,0,0.35")
## 下面这里使用加b前缀的byte对象才不会报错
# testso.d2co(b'3D_models/AX-01b_bearing_box.stl')

