#pragma once
#include "config.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

void create_meshgrid(float meshgrid[test_image_height][test_image_width][2]) {
    // Generates a coordinate grid for an image.
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) {
        meshgrid[i][j][0] = j;
        meshgrid[i][j][1] = i;
    }
}

template <int height, int width, int channels>
void convert_points_from_homogeneous(const float points[height][width][channels],
                                     float euc_points[height][width][channels-1]) {

    // Function that converts points from homogeneous to Euclidean space.
    const float eps = 1e-8;

    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        const float z = points[i][j][channels-1];
        if (abs(z) > eps) {
            for (int k = 0; k < channels-1; k++) euc_points[i][j][k] = points[i][j][k] / z;
        } else {
            for (int k = 0; k < channels-1; k++) euc_points[i][j][k] = points[i][j][k];
        }
    }
}


template <int height, int width, int channels>
void convert_points_to_homogeneous(const float xyz[height][width][channels],
                                   float padded_xyz[height][width][channels+1]) {
    // Function that converts points from Euclidean to homogeneous space.
    for (int i = 0; i < height; i++) for (int j = 0; j < width; j++) {
        for (int k = 0; k < channels; k++) padded_xyz[i][j][k] = xyz[i][j][k];
        padded_xyz[i][j][channels] = 1;
    }
}

void project_points(const float points_3d[test_image_height][test_image_width][3],
                    const float camera_matrix[3][3],
                    float points_2d[test_image_height][test_image_width][2]) {

    // Projects a 3d point onto the 2d camera plane.

    // project back using depth dividing in a safe way
    float xy_coords[test_image_height][test_image_width][2];
    convert_points_from_homogeneous<test_image_height, test_image_width, 3>(points_3d, xy_coords);

    float x_coord[test_image_height][test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) x_coord[i][j] = xy_coords[i][j][0];

    float y_coord[test_image_height][test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) y_coord[i][j] = xy_coords[i][j][1];

    // unpack intrinsics
    const float fx = camera_matrix[0][0];
    const float fy = camera_matrix[1][1];
    const float cx = camera_matrix[0][2];
    const float cy = camera_matrix[1][2];

    // apply intrinsics ans return
    float u_coord[test_image_height][test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) u_coord[i][j] = x_coord[i][j] * fx + cx;

    float v_coord[test_image_height][test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) v_coord[i][j] = y_coord[i][j] * fy + cy;

    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) {
        points_2d[i][j][0] = u_coord[i][j];
        points_2d[i][j][1] = v_coord[i][j];
    }
}


void unproject_points(const float points_2d[test_image_height][test_image_width][2],
                      const float depth[test_image_height][test_image_width],
                      const float camera_matrix[3][3],
                      float points_3d[test_image_height][test_image_width][3]) {

    // Unprojects a 2d point in 3d.
    // Transform coordinates in the pixel frame to the camera frame.

    // unpack coordinates
    float u_coord[test_image_height][test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) u_coord[i][j] = points_2d[i][j][0];

    float v_coord[test_image_height][test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) v_coord[i][j] = points_2d[i][j][1];

    // unpack intrinsics
    const float fx = camera_matrix[0][0];
    const float fy = camera_matrix[1][1];
    const float cx = camera_matrix[0][2];
    const float cy = camera_matrix[1][2];

    // projective
    float x_coord[test_image_height][test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) x_coord[i][j] = (u_coord[i][j] - cx) / fx;

    float y_coord[test_image_height][test_image_width];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) y_coord[i][j] = (v_coord[i][j] - cy) / fy;

    float xyz[test_image_height][test_image_width][2];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) {
        xyz[i][j][0] = x_coord[i][j];
        xyz[i][j][1] = y_coord[i][j];
    }

    float padded_xyz[test_image_height][test_image_width][3];
    convert_points_to_homogeneous<test_image_height, test_image_width, 2>(xyz, padded_xyz);

    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) for (int k = 0; k < 3; k++)
        points_3d[i][j][k] = padded_xyz[i][j][k] * depth[i][j];

}


void depth_to_3d(const float depth[test_image_height][test_image_width],
                 const float camera_matrix[3][3],
                 float points_3d[test_image_height][test_image_width][3]) {
    // Compute a 3d point per pixel given its depth value and the camera intrinsics.

    // create base coordinates grid
    float points_2d[test_image_height][test_image_width][2];
    create_meshgrid(points_2d);

    // project pixels to camera frame
    unproject_points(points_2d, depth, camera_matrix, points_3d);
}

void transform_points(const Matrix4f trans_01,
                      const float points_1[test_image_height][test_image_width][3],
                      float points_0[test_image_height][test_image_width][3]) {
    // Function that applies transformations to a set of points.

    // to homogeneous
    float points_1_h[test_image_height][test_image_width][4];
    convert_points_to_homogeneous<test_image_height, test_image_width, 3>(points_1, points_1_h);

    // transform coordinates
    float points_0_h[test_image_height][test_image_width][4];
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) {
        Vector4f point_1_h;
        for (int k = 0; k < 4; k++) point_1_h(k) = points_1_h[i][j][k];
        Vector4f point_0_h = trans_01 * point_1_h;
        for (int k = 0; k < 4; k++) points_0_h[i][j][k] = point_0_h(k);
    }

    // to euclidean
    convert_points_from_homogeneous<test_image_height, test_image_width, 4>(points_0_h, points_0);
}
