#include <opencv2/opencv.hpp>
#include "config.h"
#include "functional.h"

void load_image(const string image_filename, float reference_image[org_image_height][org_image_width][3]) {
    cv::Mat bgr = cv::imread(image_filename, -1);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
        reference_image[i][j][k] = rgb.data[540*3*i + 3*j + k];
}


void save_image(const string image_filename, float depth[test_image_height][test_image_width]) {
    cv::Mat gray(test_image_height, test_image_width, CV_8U);
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++)
        gray.data[test_image_width*i + j] = depth[i][j] * 150;
    cv::imwrite(image_filename, gray);
}


void PreprocessImage::apply_rgb(const float org_image[org_image_height][org_image_width][3], float image[3][test_image_height][test_image_width]) {
    float src[3][org_image_height][org_image_width];
    for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
        src[k][i][j] = org_image[i][j][k];

    interpolate<3, org_image_height, org_image_width, test_image_height, test_image_width>(src, image, "bilinear");
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) for (int k = 0; k < 3; k++) {
        image[k][i][j] = ((image[k][i][j] / scale_rgb) - mean_rgb[k]) / std_rgb[k];
    }
}


void PreprocessImage::get_updated_intrinsics(float updated_intrinsics[3][3]) {
    updated_intrinsics[0][0] = fx;
    updated_intrinsics[0][1] = 0;
    updated_intrinsics[0][2] = cx;
    updated_intrinsics[1][0] = 0;
    updated_intrinsics[1][1] = fy;
    updated_intrinsics[1][2] = cy;
    updated_intrinsics[2][0] = 0;
    updated_intrinsics[2][1] = 0;
    updated_intrinsics[2][2] = 1;
}
