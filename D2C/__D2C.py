import cv2
import ctypes
from ctypes import *
import numpy as np

class D2C():
    def __init__(self,obj_name) -> None:
        '''
            lib_path、mesh_path、cfg_path need byte type.
        '''
        lib_path = {'gear':"D2C/SO/gear/libtest_localization.so",
                    'obj_01':"D2C/SO/T_Less_obj_01/libtest_localization.so",
                    'mesh_01':"D2C/SO/mesh_01/libtest_localization.so",
                    'mesh_02':"D2C/SO/mesh_02/libtest_localization.so",
                    'mesh_03':"D2C/SO/mesh_03/libtest_localization.so",
                    'mesh_04':"D2C/SO/mesh_04/libtest_localization.so",
                    'mesh_05':"D2C/SO/mesh_05/libtest_localization.so"}  
        mesh_path = {'gear':b"D2C/3D_models/Gear.stl",
                     'obj_01':b'D2C/3D_models/obj_01.stl',
                     'mesh_01':b'D2C/3D_models/mesh_01.stl',
                    #  'mesh_02':b'D2C/3D_models/mesh_02.stl',
                     'mesh_02':b'D2C/3D_models/mesh_02a_0.5_m8_30.stl',
                    #  'mesh_03':b'D2C/3D_models/mesh_03.stl',
                    #  'mesh_03':b'D2C/3D_models/mesh_03_0.8.stl',
                     'mesh_03':b'D2C/3D_models/mesh_03_0.5.stl',
                    #  'mesh_04':b'D2C/3D_models/mesh_04_4_3.0.stl',
                     'mesh_04':b'D2C/3D_models/mesh_04_1_1.0.stl',
                     'mesh_05':b'D2C/3D_models/mesh_05.stl'}
        cfg_path  = {'gear':b"D2C/test_images/gear.yml",
                     'obj_01':b'D2C/test_images/obj_01.yml',
                     'mesh_01':b'D2C/test_images/mesh_01.yml',
                    #  'mesh_02':b'D2C/test_images/mesh_02.yml',
                    #  'mesh_02':b'D2C/test_images/mesh_02_1_m8_30.yml',
                     'mesh_02':b'D2C/test_images/mesh_02_daheng.yml',

                    #  'mesh_03':b'D2C/test_images/mesh_03.yml',
                     'mesh_03':b'D2C/test_images/mesh_03_daheng.yml',

                    #  'mesh_04':b'D2C/test_images/mesh_04_UsbCamera.yml',
                    #  'mesh_04':b'D2C/test_images/mesh_04.yml',
                     'mesh_04':b'D2C/test_images/mesh_04_daheng.yml',

                     'mesh_05':b'D2C/test_images/mesh_05_daheng.yml',
                    #  'mesh_05':b'D2C/test_images/mesh_05.yml'
                     }

        self.SO = CDLL(lib_path[obj_name])
        self.mesh_path = mesh_path[obj_name]
        self.cfg_path = cfg_path[obj_name]
        self.r_vec = [ctypes.c_double(),ctypes.c_double(),ctypes.c_double()]
        self.t_vec = [ctypes.c_double(),ctypes.c_double(),ctypes.c_double()]
        self.roi   = [ctypes.c_int(),ctypes.c_int(),ctypes.c_int(),ctypes.c_int()] # x1, x2, w, h
        self.running_flag = ctypes.c_bool()
        # self.SO.createD2CO(self.mesh_path, self.cfg_path)
    def inference_image(self,image,R_vec,t_vec):
        '''
        parameter
        ---------
            image   :   cv mat
            R_vec   :   byte type str, separate by "," , (rad) , rotation vector from R that world to camera.
            t_vec   :   byte type str, separate by "," , (m) , The vector pointing to the world coordinate system in the camera coordinate system.
        '''
        frame = image.ctypes.data_as(ctypes.c_char_p)
        run_flag = self.SO.d2co(self.mesh_path, self.cfg_path, frame, R_vec, t_vec) # test_localization_once.cpp
        return run_flag

    def inference_image_pose(self,image,image_refined_path,R_vec,t_vec):
        '''
        parameter
        ---------
            image   :   cv mat
            R_vec   :   byte type str, separate by "," , (rad) , rotation vector from R that world to camera.
            t_vec   :   byte type str, separate by "," , (m) , The vector pointing to the world coordinate system in the camera coordinate system.
        '''
        frame = image.ctypes.data_as(ctypes.c_char_p)
        run_flag = self.SO.d2co(self.mesh_path, self.cfg_path, frame, R_vec, t_vec,
                                byref(self.r_vec[0]),byref(self.r_vec[1]),byref(self.r_vec[2]),
                                byref(self.t_vec[0]),byref(self.t_vec[1]),byref(self.t_vec[2]),
                                image_refined_path) # test_localization_once.cpp
        return run_flag

    def inference_image_PoseReture(self,image,R_vec,t_vec,GT_Pose):
        '''
        parameter
        ---------
            image   :   cv mat
            R_vec   :   byte type str, separate by "," , (rad) , rotation vector from R that world to camera.
            t_vec   :   byte type str, separate by "," , (m) , The vector pointing to the world coordinate system in the camera coordinate system.
        '''
        frame = image.ctypes.data_as(ctypes.c_char_p)
        run_flag = self.SO.d2co(self.mesh_path, self.cfg_path, frame, R_vec, t_vec,
                                byref(self.r_vec[0]),byref(self.r_vec[1]),byref(self.r_vec[2]),
                                byref(self.t_vec[0]),byref(self.t_vec[1]),byref(self.t_vec[2]),
                                byref(GT_Pose))
        return run_flag

    def inference_ROI(self,ROI,R_vec:str,t_vec:str):
        '''
        parameter
        ---------
            ROI     :   cv mat roi
            R_vec   :   byte type str, separate by "," , (rad) , rotation vector from R that world to camera.
            t_vec   :   byte type str, separate by "," , (m) , The vector pointing to the world coordinate system in the camera coordinate system.
        '''
        frame = ROI.ctypes.data_as(ctypes.c_char_p)
        self.SO.d2co(self.mesh_path, self.cfg_path, frame, R_vec, t_vec)
    def inference_realtime(self,image):
        R_vec_bstr = f"{self.r_vec[0].value},{self.r_vec[1].value},{self.r_vec[2].value}".encode()
        t_vec_bstr = f"{self.t_vec[0].value},{self.t_vec[1].value},{self.t_vec[2].value}".encode()
        
        frame = image.ctypes.data_as(ctypes.c_char_p)
        run_flag = self.SO.d2co(self.mesh_path, self.cfg_path, frame, R_vec_bstr, t_vec_bstr,
                byref(self.r_vec[0]),byref(self.r_vec[1]),byref(self.r_vec[2]),
                byref(self.t_vec[0]),byref(self.t_vec[1]),byref(self.t_vec[2]),
                byref(self.running_flag)
                )
        return run_flag
    def inference_realtime_class(self,image):
        R_vec_bstr = f"{self.r_vec[0].value},{self.r_vec[1].value},{self.r_vec[2].value}".encode()
        t_vec_bstr = f"{self.t_vec[0].value},{self.t_vec[1].value},{self.t_vec[2].value}".encode()

        frame = image.ctypes.data_as(ctypes.c_char_p)
        run_flag = self.SO.inference(frame, R_vec_bstr, t_vec_bstr,
                                     byref(self.r_vec[0]),byref(self.r_vec[1]),byref(self.r_vec[2]),
                                     byref(self.t_vec[0]),byref(self.t_vec[1]),byref(self.t_vec[2]),
                                     byref(self.running_flag)
                                    )
    def inference_realtime_class_single(self,image):
        R_vec_bstr = f"{self.r_vec[0].value},{self.r_vec[1].value},{self.r_vec[2].value}".encode()
        t_vec_bstr = f"{self.t_vec[0].value},{self.t_vec[1].value},{self.t_vec[2].value}".encode()

        frame = image.ctypes.data_as(ctypes.c_char_p)
        self.roi[0].value = 290
        self.roi[1].value = 118
        self.roi[2].value = 200
        self.roi[3].value = 200
        run_flag = self.SO.inference(frame, R_vec_bstr, t_vec_bstr,
                                     byref(self.r_vec[0]),byref(self.r_vec[1]),byref(self.r_vec[2]),
                                     byref(self.t_vec[0]),byref(self.t_vec[1]),byref(self.t_vec[2]),
                                     byref(self.roi[0]),byref(self.roi[1]),byref(self.roi[2]),byref(self.roi[3]),
                                     byref(self.running_flag)
                                    )
        return run_flag