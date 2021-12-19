#include "config.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

// GEOMETRIC UTILS
void pose_distance(float reference_pose[4][4], float measurement_pose[4][4], float &combined_measure, float &R_measure, float &t_measure) {
    // :param reference_pose: 4x4 numpy array, reference frame camera-to-world pose (not extrinsic matrix!)
    // :param measurement_pose: 4x4 numpy array, measurement frame camera-to-world pose (not extrinsic matrix!)
    // :return combined_measure: float, combined pose distance measure
    // :return R_measure: float, rotation distance measure
    // :return t_measure: float, translation distance measure

    Matrix4f r_pose, m_pose;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r_pose(i, j) = reference_pose[i][j];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) m_pose(i, j) = measurement_pose[i][j];

    Matrix4f rel_pose = r_pose.inverse() * m_pose;

    Matrix3f R;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R(i, j) = rel_pose(i, j);
    Vector3f t;
    for (int i = 0; i < 3; i++) t(i) = rel_pose(i, 3);

    R_measure = sqrt(2 * (1 - min((float) 3.0, (float) R.trace()) / 3));
    t_measure = t.norm();
    combined_measure = sqrt(t_measure * t_measure + R_measure * R_measure);

    // rel_pose = np.dot(np.linalg.inv(reference_pose), measurement_pose)
    // R = rel_pose[:3, :3]
    // t = rel_pose[:3, 3]
    // R_measure = np.sqrt(2 * (1 - min(3.0, np.matrix.trace(R)) / 3))
    // t_measure = np.linalg.norm(t)
    // combined_measure = np.sqrt(t_measure ** 2 + R_measure ** 2)
}


void get_warp_grid_for_cost_volume_calculation(float warp_grid[3][warp_grid_width * warp_grid_height]) {
    for (int i = 0; i < warp_grid_height; i++) for (int j = 0; j < warp_grid_width; j++) warp_grid[0][i * warp_grid_height + j] = j;
    for (int i = 0; i < warp_grid_height; i++) for (int j = 0; j < warp_grid_width; j++) warp_grid[1][i * warp_grid_height + j] = i;
    for (int i = 0; i < warp_grid_height; i++) for (int j = 0; j < warp_grid_width; j++) warp_grid[2][i * warp_grid_height + j] = 1;
}


bool is_pose_available(float pose[4][4]) {
    // is_nan = np.isnan(pose).any()
    // is_inf = np.isinf(pose).any()
    // is_neg_inf = np.isneginf(pose).any()
    // if is_nan or is_inf or is_neg_inf:
    //     return False
    // else:
    //     return True
    return true;
}