#include "config.h"

pair<float[4][4], float[org_image_height][org_image_width][3]> encode_buf_pair(float pose[4][4], float image[org_image_height][org_image_width][3]) {
    pair<float[4][4], float[org_image_height][org_image_width][3]> ret;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) ret.first[i][j] = pose[i][j];
    for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
        ret.second[i][j][k] = image[i][j][k];
    return ret;
}


void decode_buf_pair(pair<float[4][4], float[org_image_height][org_image_width][3]> buf, float pose[4][4], float image[org_image_height][org_image_width][3]) {
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) pose[i][j] = buf.first[i][j];
    for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
        image[i][j][k] = buf.second[i][j][k];
}


float KeyframeBuffer::calculate_penalty(float t_score, float R_score) {
    float degree = 2.0;
    float R_penalty = pow(abs(R_score - optimal_R_score), degree);
    float t_diff = t_score - optimal_t_score;
    float t_penalty;
    if (t_diff < 0.0) {
        t_penalty = 5.0 * pow(abs(t_diff), degree);
    } else {
        t_penalty = pow(abs(t_diff), degree);
    }
    return R_penalty + t_penalty;
}


int KeyframeBuffer::try_new_keyframe(float pose[4][4], float image[org_image_height][org_image_width][3]) {
    if (is_pose_available(pose)) {
        __tracking_lost_counter = 0;
        if (buffer.empty()) {
            buffer.push_back(encode_buf_pair(pose, image));
            return 0;  // pose is available, new frame added but buffer was empty, this is the first frame, no depth map prediction will be done
        } else {
            float last_pose[4][4];
            float last_image[org_image_height][org_image_width][3];
            decode_buf_pair(buffer.back(), last_pose, last_image);

            float combined_measure, R_measure, t_measure;
            pose_distance(pose, last_pose, combined_measure, R_measure, t_measure);

            if (combined_measure >= keyframe_pose_distance) {
                buffer.push_back(encode_buf_pair(pose, image));
                return 1;  // pose is available, new frame added, everything is perfect, and we will predict a depth map later
            } else {
                return 2;  // pose is available but not enough change has happened since the last keyframe
            }
        }
    } else {
        __tracking_lost_counter += 1;
        if (__tracking_lost_counter > 30) {
            if (!buffer.empty()) {
                buffer.clear();
                return 3;  // a pose reading has not arrived for over a second, tracking is now lost
            } else {
                return 4;  // we are still very lost
            }
        } else {
            return 5;  // pose is not available right now, but not enough time has passed to consider lost, there is still hope :)
        }
    }
}


int KeyframeBuffer::get_best_measurement_frames(pair<float[4][4], float[org_image_height][org_image_width][3]> measurement_frames[test_n_measurement_frames]) {
    float reference_pose[4][4];
    float reference_image[org_image_height][org_image_width][3];
    decode_buf_pair(buffer.back(), reference_pose, reference_image);

    const int len_buf = buffer.size();
    const int n_requested_measurement_frames = min(test_n_measurement_frames, len_buf - 1);

    pair<float, int> penalties[len_buf - 1];
    for (int i = 0; i < len_buf - 1; i++) {
        pair<float[4][4], float[org_image_height][org_image_width][3]> buf = buffer.at(i);
        float measurement_pose[4][4];
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) measurement_pose[i][j] = buf.first[i][j];

        float combined_measure, R_measure, t_measure;
        pose_distance(reference_pose, measurement_pose, combined_measure, R_measure, t_measure);

        float penalty = calculate_penalty(t_measure, R_measure);
        penalties[i] = pair<float, int>(penalty, i);
    }
    sort(penalties, penalties + len_buf - 1);
    // indices = np.argpartition(penalties, n_requested_measurement_frames - 1)[:n_requested_measurement_frames]

    for (int f = 0; f < n_requested_measurement_frames; f++) {
        // measurement_frames[i] = buffer.at(penalties[i].second);
        pair<float[4][4], float[org_image_height][org_image_width][3]> buf = buffer.at(penalties[f].second);
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) measurement_frames[f].first[i][j] = buf.first[i][j];
        for (int i = 0; i < org_image_height; i++) for (int j = 0; j < org_image_width; j++) for (int k = 0; k < 3; k++)
            measurement_frames[f].second[i][j][k] = buf.second[i][j][k];
    }
    return n_requested_measurement_frames;
}