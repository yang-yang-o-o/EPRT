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
        return self.cut_img
if __name__ == "__main__":
    img = image_cut().cut(cv2.imread('222.png'))
    cv2.imshow('1',img)
    cv2.waitKey(0)