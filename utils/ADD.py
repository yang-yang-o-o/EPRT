import numpy as np

def ADD_caculate(pointclond,pose_pr,pose_gt):
    '''
    Input
    -----
        pointclond:     3D points in object coordinate system, 4xn
        pose_pr:        4x4
        pose_gt:        4x4
    Output
    ------
        ADD:            Average Dot Distance
    '''
    point_c_pr = pose_pr.dot(pointclond)[:3,:].T    # nx3
    point_c_gt = pose_gt.dot(pointclond)[:3,:].T    # nx3
    norm         = np.linalg.norm(point_c_gt - point_c_pr, axis=1)
    ADD          = np.mean(norm)
    return ADD

def ADD_S_caculate(pointclond,pose_pr,pose_gt):
    '''
    Input
    -----
        pointclond:     3D points in object coordinate system, 4xn
        pose_pr:        4x4
        pose_gt:        4x4
    Output
    ------
        ADD_S:            Average Dot Distance Symmetry 
    '''
    point_c_pr = pose_pr.dot(pointclond)[:3,:].T    # nx3
    point_c_gt = pose_gt.dot(pointclond)[:3,:].T    # nx3
    total_dist = 0
    for i in range(point_c_gt.shape[0]):
        norm = np.linalg.norm(point_c_pr - point_c_gt[i], axis=1)
        min_id = np.argmin(norm)
        point_dist = np.linalg.norm(point_c_pr[min_id] - point_c_gt[i])
        total_dist += point_dist
    ADD_S = total_dist/point_c_gt.shape[0]
    return ADD_S