#include "config.h"
#include "model.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

// GEOMETRIC UTILS
void pose_distance(const float reference_pose[4][4], const float measurement_pose[4][4], float &combined_measure, float &R_measure, float &t_measure) {
    // :param reference_pose: 4x4 numpy array, reference frame camera-to-world pose (not extrinsic matrix!)
    // :param measurement_pose: 4x4 numpy array, measurement frame camera-to-world pose (not extrinsic matrix!)
    // :return combined_measure: float, combined pose distance measure
    // :return R_measure: float, rotation distance measure
    // :return t_measure: float, translation distance measure

    Matrix4f r_pose, m_pose;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) r_pose(i, j) = reference_pose[i][j];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) m_pose(i, j) = measurement_pose[i][j];

    Matrix4f rel_pose = r_pose.inverse() * m_pose;

    // Matrix3f R;
    // for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R(i, j) = rel_pose(i, j);
    // Vector3f t;
    // for (int i = 0; i < 3; i++) t(i) = rel_pose(i, 3);

    Matrix3f R = rel_pose.block(0, 0, 3, 3);
    Vector3f t = rel_pose.block(0, 3, 3, 1);

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

void calculate_cost_volume_by_warping(const float image1[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)],
                                      const float image2[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)],
                                      const float ls_pose1[4][4],
                                      const float ls_pose2[4][4],
                                      const float ls_K[3][3],
                                      const float ls_warp_grid[3][warp_grid_width * warp_grid_height],
                                      float cost_volume[n_depth_levels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)]) {

    const int channels = fe1_out_channels;
    const int height = fe1_out_size(test_image_height);
    const int width = fe1_out_size(test_image_width);

    Matrix4f pose1, pose2;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pose1(i, j) = ls_pose1[i][j];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pose2(i, j) = ls_pose2[i][j];

    Matrix4f extrinsic2 = pose2.inverse() * pose1;

    // Matrix3f R;
    // for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R(i, j) = extrinsic2(i, j);
    // Vector3f t;
    // for (int i = 0; i < 3; i++) t(i) = extrinsic2(i, 3);

    Matrix3f R = extrinsic2.block(0, 0, 3, 3);
    Vector3f t = extrinsic2.block(0, 3, 3, 1);

    Matrix3f K;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) K(i, j) = ls_K[i][j];
    MatrixXf warp_grid(3, warp_grid_width * warp_grid_height);
    for (int i = 0; i < 3; i++) for (int j = 0; j < warp_grid_width * warp_grid_height; j++) warp_grid(i, j) = ls_warp_grid[i][j];

    Vector3f _Kt = K * t;
    Matrix3f K_R_Kinv = K * R * K.inverse();
    MatrixXf K_R_Kinv_UV(3, warp_grid_width * warp_grid_height);
    K_R_Kinv_UV = K_R_Kinv * warp_grid;

    MatrixXf Kt(3, warp_grid_width * warp_grid_height);
    for (int i = 0; i < warp_grid_width * warp_grid_height; i++) Kt.block(0, i, 3, 1) = _Kt;

    const float inverse_depth_base = 1.0 / max_depth;
    const float inverse_depth_step = (1.0 / min_depth - 1.0 / max_depth) / (n_depth_levels - 1);

    const float width_normalizer = width / 2.0;
    const float height_normalizer = height / 2.0;

    for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
        cost_volume[depth_i][j][k] = 0;

    for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) {
        const float this_depth = 1.0 / (inverse_depth_base + depth_i * inverse_depth_step);

        MatrixXf _warping(warp_grid_width * warp_grid_height, 3);
        _warping = (K_R_Kinv_UV + (Kt / this_depth)).transpose();

        MatrixXf _warping0(warp_grid_width * warp_grid_height, 2);
        VectorXf _warping1(warp_grid_width * warp_grid_height);
        _warping0 = _warping.block(0, 0, warp_grid_width * warp_grid_height, 2);
        _warping1 = _warping.block(0, 2, warp_grid_width * warp_grid_height, 1).array() + 1e-8f;

        _warping0.block(0, 0, warp_grid_width * warp_grid_height, 1).array() /= _warping1.array();
        _warping0.block(0, 0, warp_grid_width * warp_grid_height, 1).array() -= width_normalizer;
        _warping0.block(0, 0, warp_grid_width * warp_grid_height, 1).array() /= width_normalizer;

        _warping0.block(0, 1, warp_grid_width * warp_grid_height, 1).array() /= _warping1.array();
        _warping0.block(0, 1, warp_grid_width * warp_grid_height, 1).array() -= height_normalizer;
        _warping0.block(0, 1, warp_grid_width * warp_grid_height, 1).array() /= height_normalizer;

        float warping[warp_grid_height][warp_grid_width][2];
        for (int i = 0; i < warp_grid_height; i++) for (int j = 0; j < warp_grid_width; j++) for (int k = 0; k < 2; k++)
            warping[i][j][k] = _warping0(warp_grid_width * i + j, k);

        float warped_image2[channels][height][width];
        grid_sample(image2, warping, warped_image2);

        for (int i = 0; i < channels; i++) for (int j = 0; j < height; j++) for (int k = 0; k < width; k++)
            cost_volume[depth_i][j][k] += (image1[i][j][k] * warped_image2[i][j][k]) / channels;

    }
}

void cost_volume_fusion(const float image1[fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)],
                        const float image2s[test_n_measurement_frames][fe1_out_channels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)],
                        const float pose1[4][4],
                        const float pose2s[test_n_measurement_frames][4][4],
                        const float K[3][3],
                        const float warp_grid[3][warp_grid_width * warp_grid_height],
                        const int n_measurement_frames,
                        float fused_cost_volume[n_depth_levels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)]) {

    for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) for (int j = 0; j < fe1_out_size(test_image_height); j++) for (int k = 0; k < fe1_out_size(test_image_width); k++)
        fused_cost_volume[depth_i][j][k] = 0;

    for (int m = 0; m < n_measurement_frames; m++) {
        float cost_volume[n_depth_levels][fe1_out_size(test_image_height)][fe1_out_size(test_image_width)];
        calculate_cost_volume_by_warping(image1, image2s[m], pose1, pose2s[m], K, warp_grid, fused_cost_volume);
        for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) for (int j = 0; j < fe1_out_size(test_image_height); j++) for (int k = 0; k < fe1_out_size(test_image_width); k++)
            fused_cost_volume[depth_i][j][k] += cost_volume[depth_i][j][k];
    }

    for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) for (int j = 0; j < fe1_out_size(test_image_height); j++) for (int k = 0; k < fe1_out_size(test_image_width); k++)
        fused_cost_volume[depth_i][j][k] /= n_measurement_frames;

}

bool is_pose_available(const float pose[4][4]) {
    // is_nan = np.isnan(pose).any()
    // is_inf = np.isinf(pose).any()
    // is_neg_inf = np.isneginf(pose).any()
    // if is_nan or is_inf or is_neg_inf:
    //     return False
    // else:
    //     return True
    return true;
}