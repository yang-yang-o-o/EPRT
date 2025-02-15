import subprocess
import os

rx,ry,rz = 0,0,1
tx,ty,tz = 0,0,0.35

main = "./test_localization\
      3D_models/AX-01b_bearing_box.stl\
      test_images/test_camera_calib.yml\
      test_images/2.png\
      {},{},{}\
      {},{},{}".format(rx,ry,rz,tx,ty,tz)

os.system(main)