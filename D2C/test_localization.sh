#!/bin/bash

export LD_LIBRARY_PATH="LD_LIBRARY_PATH:./lib/"
#export LD_LIBRARY_PATH="LD_LIBRARY_PATH:./lib/opencv/"
#export LD_LIBRARY_PATH="LD_LIBRARY_PATH:./lib/pcl/"
#export LD_LIBRARY_PATH="LD_LIBRARY_PATH:./lib/vtk/"

./test_localization\
     -m 3D_models/AX-01b_bearing_box.stl\
     -c test_images/test_camera_calib.yml\
     -i test_images/2.png\
     -r 0,0,1\
     -t 0,0,0.35
