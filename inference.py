import json
import os
import sys

sys.path.append(os.getcwd())
import cv2
import numpy as np

import YOLOv4_tiny.__Detect as Detect
import AAE.__AAE as AAE
import D2C.__D2C as D2C

class Estimator():
    def __init__(self,obj_id:int) -> None:
        '''
        obj_id
            0   :   obj_01
            1   :   gear
            2   :   mesh_01
            3   :   mesh_02
            4   :   mesh_03
            5   :   mesh_04
            6   :   mesh_05
        '''
        # need to set
        self.obj_id = obj_id
        self.obj_name = {0:'obj_01' ,
                    1:'gear',
                    2:'mesh_01',
                    3:'mesh_02',
                    4:'mesh_03',
                    5:'mesh_04',
                    6:'mesh_05'}
        self.K_test = np.array([(640+480)/2., 0, 640/2, 0, (640+480)/2., 480/2, 0, 0, 1]).reshape(3,3)
        # self.K_test = np.array([1643.1, 0, 662.9, 0, 1643.8, 500.8, 0, 0, 1]).reshape(3,3)

        self._Detect = Detect.Detect(self.obj_name[self.obj_id],use_cuda=False)
        self._AAE = AAE.AAE(self.obj_name[self.obj_id],train=False)
        self._D2C = None
        try:
            self._D2C = D2C.D2C(self.obj_name[self.obj_id])
        except:
            print("D2C need linux platform")
        else:
            print("D2C created")
        
    def inference(self,image,detect_debug=True,AAE_debug=True):
        '''
        Input
        -----
        image : cv mat

        Output
        ------
        T : refined pose
        '''
        # detect
        self._Detect.boxes = self._Detect.detect(image)[:,0:1,:] # (B,1,7)
        if detect_debug:
            image = self._Detect.plot()
            crop_img, _ = self._Detect.crop()
            cv2.imshow('image',image)
            cv2.imshow('crop_img',crop_img[0])
            print(f"image shape: {image.shape}",
                  f"crop image shape: {crop_img[0].shape}",
                  sep='\n')
            key = cv2.waitKey(0)
            if key == 27:
                return
        crop_img , predict_bbx = self._Detect.crop()

        # AAE
        T = self._AAE.inference(crop_img[0] , self.K_test , predict_bbx[0])
        if AAE_debug:
            self._AAE.reconstruct(crop_img[0])
        R_vec = cv2.Rodrigues(T[:3,:3])[0].transpose()
        t_vec = T[:3,3].transpose()/1000.0

        # D2C
        if self._D2C is not None:
            R_vec_bstr = f"{R_vec[0][0]},{R_vec[0][1]},{R_vec[0][2]}".encode()
            t_vec_bstr = f"{t_vec[0]},{t_vec[1]},{t_vec[2]}".encode()
            self._D2C.inference_image(image,R_vec_bstr,t_vec_bstr)

    def inference_for_each_bbx(self,image,detect_debug=True,AAE_debug=True,use_gtbbox=[False,None]):
        '''
        Input
        -----
        image : cv mat

        Output
        ------
        T : refined pose
        '''
        # self._Detect.boxes = self._Detect.detect(image)[:,0:1,:] # (B,1,7)
        if use_gtbbox[0]:
            boxes = np.array(use_gtbbox[1])
            self._Detect.image = image.copy()
        else: 
            boxes = self._Detect.detect(image).copy()
        for i in range(boxes.shape[1]):
            ## detect
            self._Detect.boxes = boxes[:,i:i+1,:]
            if detect_debug:
                _image = self._Detect.plot()
                crop_img, _ ,_ = self._Detect.crop()
                cv2.imshow('image',_image)
                cv2.imshow('crop_img',crop_img[0])
                print(f"image shape: {_image.shape}",
                    f"crop image shape: {crop_img[0].shape}",
                    sep='\n')
                if self._D2C is None:
                    key = cv2.waitKey(0)
                    if key == 27:
                        return key

            crop_img , predict_bbx , predict_bbx_rectify= self._Detect.crop()
            print(predict_bbx)
            
            # AAE
            T = self._AAE.inference(crop_img[0] , self.K_test , predict_bbx[0])
            if self._D2C is None and AAE_debug:
                self._AAE.reconstruct(crop_img[0])
            R_vec = cv2.Rodrigues(T[:3,:3])[0].transpose()
            t_vec = T[:3,3].transpose()/1000.0
            print(R_vec)
            print(t_vec)
            ### AABB
            npy_path = self._AAE.data_path[self.obj_name[self.obj_id]]+'/'+self._AAE.data_path[self.obj_name[self.obj_id]].split('/')[-1]+".npy"
            
            if self._D2C is None and not os.path.exists(npy_path):
                pointcloud = npy_path.replace('.npy','.ply')
                corner3D = self._AAE.get_AABB_bbx(pointcloud)
                np.save(npy_path,corner3D)
            else:
                corner3D = np.load(npy_path)
            corner2D = self._AAE.ProjectCorner3D(_image,corner3D,T,self.K_test,show_point=True,show_edge=True,color=(255,0,0))
            cv2.imshow('image_project',_image)
            if self._D2C is None:
                key = cv2.waitKey(0)
                if key == 27:
                    return key
            ###

            # D2C
            run_flag = None
            if self._D2C is not None:
                R_vec_bstr = f"{R_vec[0][0]},{R_vec[0][1]},{R_vec[0][2]}".encode()
                t_vec_bstr = f"{t_vec[0]},{t_vec[1]},{t_vec[2]}".encode()
                run_flag = self._D2C.inference_image(image,R_vec_bstr,t_vec_bstr)
                # run_flag = ctypes.string_at(run_flag, -1).decode("utf-8")
                print(run_flag)
            if run_flag == 2:
                return run_flag

    def inference_for_each_bbx_for_mesh_02(self,image,detect_debug=True,AAE_debug=True):
        '''
        Input
        -----
        image : cv mat

        Output
        ------
        T : refined pose
        '''
        # self._Detect.boxes = self._Detect.detect(image)[:,0:1,:] # (B,1,7)
        boxes = self._Detect.detect(image).copy()
        # for i in range(boxes.shape[1]):
        for i in range(1):
            ## detect
            # self._Detect.boxes = boxes[:,i:i+1,:]
            self._Detect.boxes = np.array([[[100/640,100/480,550/640,400/480,1,1,0]]]) # for mesh_02_1_m8_30
            if detect_debug:
                image = self._Detect.plot()
                crop_img, _ = self._Detect.crop()
                cv2.imshow('image',image)
                cv2.imshow('crop_img',crop_img[0])
                print(f"image shape: {image.shape}",
                    f"crop image shape: {crop_img[0].shape}",
                    sep='\n')
                if self._D2C is None:
                    key = cv2.waitKey(0)
                    if key == 27:
                        return
            crop_img , predict_bbx = self._Detect.crop()
            print(predict_bbx)

            # AAE
            T = self._AAE.inference(crop_img[0] , self.K_test , predict_bbx[0])
            if AAE_debug:
                self._AAE.reconstruct(crop_img[0])
            R_vec = cv2.Rodrigues(T[:3,:3])[0].transpose()
            t_vec = T[:3,3].transpose()/1000.0

            # D2C
            run_flag = None
            if self._D2C is not None:
                R_vec_bstr = f"{R_vec[0][0]},{R_vec[0][1]},{R_vec[0][2]}".encode()
                t_vec_bstr = f"{t_vec[0]},{t_vec[1]},{t_vec[2]}".encode()
                run_flag = self._D2C.inference_image(image,R_vec_bstr,t_vec_bstr)
                # run_flag = ctypes.string_at(run_flag, -1).decode("utf-8")
                print(run_flag)
            if run_flag == "2":
                return

    def inference_realtime(self,AAE_debug=False):
        # detect
        if self._D2C is None:
            raise EnvironmentError("linux platform needed!")
        cap = cv2.VideoCapture(0)
        initial = True
        while True:
            ret, image = cap.read()
            self._Detect.boxes = self._Detect.detect(image)[:,1:2,:] # (B,1,7)
            if initial:
                image = self._Detect.plot()
                crop_img, _ = self._Detect.crop()
                cv2.imshow('image',image)
                cv2.imshow('crop_img',crop_img[0])
                print(f"image shape: {image.shape}",
                    f"crop image shape: {crop_img[0].shape}",
                    sep='\n')
                key = cv2.waitKey(0)
                if key == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            crop_img , predict_bbx = self._Detect.crop()
            print(predict_bbx)

            # AAE
            if initial:
                T = self._AAE.inference(crop_img[0] , self.K_test , predict_bbx[0])
                if AAE_debug:
                    self._AAE.reconstruct(crop_img[0])
                R_vec = cv2.Rodrigues(T[:3,:3])[0].transpose()
                t_vec = T[:3,3].transpose()/1000.0

                self._D2C.r_vec[0].value = R_vec[0][0]; self._D2C.r_vec[1].value = R_vec[0][1]; self._D2C.r_vec[2].value = R_vec[0][2]
                self._D2C.t_vec[0].value = t_vec[0]; self._D2C.t_vec[1].value = t_vec[1]; self._D2C.t_vec[2].value = t_vec[2]
                self._D2C.running_flag.value = initial
            # D2C
            run_flag = self._D2C.inference_realtime_class_single(image)
            initial = self._D2C.running_flag.value
            if run_flag == 2:
                cap.release()
                return
        
if __name__ == "__main__":
    #########################################################################################
    ################################### inference for image #################################
    id = 5
    image_id = 45

    pose_estimator = Estimator(obj_id=id)
    # # # image = cv2.imread('data/image/image_0.png')
    # # image = cv2.imread(f'data/image/{pose_estimator.obj_name[id]}_real/image_{image_id}.jpg')
    # # image = cv2.imread(f'data/image/{pose_estimator.obj_name[id]}/18.png')  # mesh_02
    # # image = cv2.imread(f'data/image/{pose_estimator.obj_name[id]}_real/18.png')  # mesh_06
    image = cv2.imread('data/2/1.png')
    # image = cv2.imread(f"/mnt/hgfs/training_data/{pose_estimator.obj_name[id]}/image/image_{image_id}.png")
    # # image = cv2.imread(f"/mnt/hgfs/Release/data/image/{pose_estimator.obj_name[id]}_real/image_{image_id}.jpg") 
    # #                         # nesh_01:11,20,22,30,31; mesh_02:13; mesh_03:all 
    # # # image = cv2.imread(f"F:\AAE_D2C_YOLO\YOLOv4-tiny\YOLO-tiny\\training_data\{pose_estimator.obj_name[id]}\image\image_4.png")
    # # # pose_estimator.inference(image,detect_debug=True,AAE_debug=True)
    pose_estimator.inference_for_each_bbx(image,detect_debug=True,AAE_debug=True)
    #########################################################################################

    #########################################################################################
    ################################### inference for realtime ##############################
    # id = 5
    # pose_estimator = Estimator(obj_id=id)
    # pose_estimator.inference_realtime()
    #########################################################################################