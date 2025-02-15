from math import *
import numpy as np

def rotation_k_θ(k,θ):
    '''
    input:
        k   :   [x,y,z], symmetry_axis
        θ   :   rad, rotation angle
    output:
        R   :   rotation matrix R(k,θ)
    '''
    return np.array([[k[0]*k[0]*(1-cos(θ))+cos(θ)     , k[1]*k[0]*(1-cos(θ))-k[2]*sin(θ), k[2]*k[0]*(1-cos(θ))+k[1]*sin(θ)],
                        [k[0]*k[1]*(1-cos(θ))+k[2]*sin(θ), k[1]*k[1]*(1-cos(θ))+cos(θ)     , k[2]*k[1]*(1-cos(θ))-k[0]*sin(θ)],
                        [k[0]*k[2]*(1-cos(θ))-k[1]*sin(θ), k[1]*k[2]*(1-cos(θ))+k[0]*sin(θ), k[2]*k[2]*(1-cos(θ))+cos(θ)]])

symmetry_axis = [1/sqrt(2),0.,1/sqrt(2)]

rotation_matrix_around_symmetry_axis = [rotation_k_θ(symmetry_axis,i*pi/180.).T for i in range(0,360,180)]

D2C_R = None; D2C_t = None  # D2C 优化后的结果，4x4

Rs = []
for R in rotation_matrix_around_symmetry_axis:
    Rs.append(np.dot(D2C_R, R))

T1 = np.concatenate(Rs[0],D2C_t,axis=1) # 3x4
T2 = np.concatenate(Rs[1],D2C_t,axis=1) # 3x4

