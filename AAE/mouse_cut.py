import cv2
class image_cut():
    def __init__(self,detect=None):
        self.img = None
        self.point1 = None
        self.point2 = None
        self.cut_img = None
        self.detect = detect
    def on_mouse(self,event, x, y, flags, param):
        img2 = self.img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
            self.point1 = (x,y)
            cv2.circle(img2, self.point1, 10, (0,255,0), 2)
            cv2.imshow('image', img2)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):               #按住左键拖曳
            cv2.rectangle(img2, self.point1, (x,y), (255,0,0), 2)
            cv2.imshow('image', img2)
        elif event == cv2.EVENT_LBUTTONUP:         #左键释放
            self.point2 = (x,y)
            cv2.rectangle(img2, self.point1, self.point2, (0,0,255), 2) 
            cv2.imshow('image', img2)
            min_x = min(self.point1[0],self.point2[0])     
            min_y = min(self.point1[1],self.point2[1])
            width = abs(self.point1[0] - self.point2[0])
            height = abs(self.point1[1] -self.point2[1])
            self.cut_img = self.img[min_y:min_y+height, min_x:min_x+width]
            cv2.destroyWindow('cut_img')
            cv2.destroyWindow('fast_true.png')
            cv2.imshow('cut_img', self.cut_img)
            if self.detect is not None:
                self.detect(self.cut_img)
    def cut(self,init_img):
        self.img = init_img
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.on_mouse)
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        center = ((self.point1[0]+self.point2[0])//2,(self.point1[1]+self.point2[1])//2)
        cv2.circle(init_img,center,2,(0,0,255),1)
        cv2.imshow('center',init_img)
        cv2.waitKey(0)
        return self.cut_img, center
    def R2rpy(self,R):
        import numpy as np
        beta_y = np.arctan2(-R[2][0],np.sqrt(np.sum(np.square(R[0][0])+np.square(R[1][0]))))*180.0/np.pi
        alpha_z = np.arctan2(R[1][0],R[0][0])*180.0/np.pi
        gamma_x = np.arctan2(R[2][1],R[2][2])*180.0/np.pi
        return gamma_x , beta_y , alpha_z
    def imgpoont2t(self,img_point,z):
        fx = 909.3773280487426
        fy = 909.3773280487426
        u0 = 642.9053428238576
        v0 = 358.0199195001692
        x = (img_point[0] - u0)/fx
        y = (img_point[1] - v0)/fy
        return (x*z,y*z,1*z)
if __name__ == "__main__":
    img = image_cut().cut(cv2.imread('222.png'))
    cv2.imshow('1',img)
    cv2.waitKey(0)