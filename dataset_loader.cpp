#include <opencv2/opencv.hpp>
#include "config.h"

void load_image(string image_filename, float reference_image[org_image_height][org_image_width][3]) {
    cv::Mat bgr = cv::imread(image_filename, -1);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
        reference_image[i][j][k] = rgb.data[org_image_width*3*i + 3*j + k];
}

// void save_img(vii img, const string name) {
//     cv::Mat opt(height, width, CV_8U);
//     for (us px = 0; px < height; px++) {
//         for (us py = 0; py < width; py++) {
//             opt.data[px*width + py] = img[px][py];
//         }
//     }
//     string fname = name + ".png";
//     cv::imwrite(fname, opt);
// }

void PreprocessImage::apply_rgb(float image[org_image_height][org_image_width][3], float image_torch[3][test_image_height][test_image_width]) {
    cv::Mat img(org_image_height, org_image_width, CV_8UC3);
    for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
        img.data[org_image_width*3*i + 3*j + k] = image[i][j][k];
    cv::resize(img, img, cv::Size(test_image_width, test_image_height), 0, 0, cv::INTER_LINEAR);

    for (int i = 0; i < test_image_height; i++) for (int j = 0; j < test_image_width; j++) for (int k = 0; k < 3; k++) {
        float val = img.data[test_image_width*3*i + 3*j + k];
        image_torch[k][i][j] = ((val / scale_rgb) - mean_rgb[k]) / std_rgb[k];
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

