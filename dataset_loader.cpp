#include <opencv2/opencv.hpp>
#include "config.h"
#include "functional.h"

void get_updated_intrinsics(const float K[3][3], float updated_intrinsics[3][3]) {
    float factor_x = (float) test_image_width / (float) org_image_width;
    float factor_y = (float) test_image_height / (float) org_image_height;

    updated_intrinsics[0][0] = K[0][0] * factor_x;
    updated_intrinsics[0][1] = 0;
    updated_intrinsics[0][2] = K[0][2] * factor_x;
    updated_intrinsics[1][0] = 0;
    updated_intrinsics[1][1] = K[1][1] * factor_y;
    updated_intrinsics[1][2] = K[1][2] * factor_y;
    updated_intrinsics[2][0] = 0;
    updated_intrinsics[2][1] = 0;
    updated_intrinsics[2][2] = 1;
}


void apply_rgb(float*** org_image, float image[3][test_image_height][test_image_width]) {
    interpolate<3, org_image_height, org_image_width, test_image_height, test_image_width>(org_image, image);
    for (int i = 0; i < 3; i++) for (int j = 0; j < test_image_height; j++) for (int k = 0; k < test_image_width; k++)
        image[i][j][k] = ((image[i][j][k] / scale_rgb) - mean_rgb[i]) / std_rgb[i];
}


void load_image(const string image_filename, float reference_image[3][test_image_height][test_image_width]) {
    cv::Mat bgr = cv::imread(image_filename, -1);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    float ***org_reference_image = new float**[3];
    new_3d(org_reference_image, 3, org_image_height, org_image_width);
    for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
        org_reference_image[k][i][j] = rgb.data[540*3*i + 3*j + k];

    apply_rgb(org_reference_image, reference_image);
    delete_3d(org_reference_image, 3, org_image_height, org_image_width);
}


void save_image(const string image_filename, float depth[test_image_height][test_image_width]) {
    cv::Mat gray(test_image_height, test_image_width, CV_8U);
    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++)
        gray.data[test_image_width*i + j] = depth[i][j] * 150;
    cv::imwrite(image_filename, gray);
}
