# const int warp_grid_width = test_image_width / 2;
# const int warp_grid_height = test_image_height / 2;
# // const int warp_grid_size = warp_grid_width * warp_grid_height;

class Config:
    # training settings
    # train_image_width = 256
    # train_image_height = 256
    train_min_depth = 0.25
    train_max_depth = 20.0
    train_n_depth_levels = 64
    # train_minimum_pose_distance = 0.125
    # train_maximum_pose_distance = 0.325
    # train_crawl_step = 3
    # train_subsequence_length = None
    # train_predict_two_way = None
    # train_freeze_batch_normalization = False
    # train_data_pipeline_workers = 8
    # train_epochs = 100000
    # train_print_frequency = 5000
    # train_validate = True
    # train_seed = int(round(time.time()))

    # test settings
    org_image_width = 540
    org_image_height = 360
    # test_image_width = 320
    # test_image_height = 256
    test_image_width = 96
    test_image_height = 64

    # org_image_width = 128
    # org_image_height = 128
    # test_image_width = 96
    # test_image_height = 64

    test_distortion_crop = 0
    test_perform_crop = False
    # test_visualize = True
    test_visualize = False
    test_n_measurement_frames = 2
    test_keyframe_buffer_size = 30
    test_keyframe_pose_distance = 0.1
    test_optimal_t_measure = 0.15
    test_optimal_R_measure = 0.0

    # SET THESE: TRAINING FOLDER LOCATIONS
    # dataset = "/home/nhsmt1123/master-thesis/deep-video-mvs/data/7scenes"
    # train_run_directory = "/home/nhsmt1123/master-thesis/deep-video-mvs/training-runs"

    # fusionnet_train_weights = "weights"
    fusionnet_test_weights = "/home/nhsmt1123/master-thesis/deep-video-mvs/dvmvs/fusionnet/weights"
    # fusionnet_train_weights = "20211129-133714"
    # fusionnet_test_weights = "fusionnet-20211202-103411"

    # SET THESE: TESTING FOLDER LOCATIONS
    # for run-testing-online.py (evaluate a single scene, WITHOUT keyframe indices, online selection)
    test_online_scene_path = "/home/nhsmt1123/master-thesis/deep-video-mvs/sample-data/hololens-dataset/000"
    # test_online_scene_path = "/home/nhsmt1123/master-thesis/deep-video-mvs/data/7scenes/chess-seq-01"

    # for run-testing.py (evaluate all available scenes, WITH pre-calculated keyframe indices)
    # test_offline_data_path = "/home/nhsmt1123/master-thesis/deep-video-mvs/sample-data"

    # below give a dataset name like tumrgbd, i.e. folder or None
    # if None, all datasets will be evaluated given that
    # their keyframe index files are in Config.test_offline_data_path/indices folder
    # test_dataset_name = "hololens-dataset"  # or None
    test_dataset_name = "hololens"  # or None

    save_path = "/home/nhsmt1123/master-thesis/dvmvs-cpp/results-py"

    # test_result_folder = "/home/nhsmt1123/master-thesis/deep-video-mvs/results/"

    # n_test_frames = 50
    n_test_frames = 373
