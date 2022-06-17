import os
from path import Path
import struct
import shutil

import cv2

def convert_hololens():
    image_dir = Path("/home/nhsmt1123/master-thesis/deep-video-mvs/sample-data/hololens-dataset/000/images/")

    save_dir = os.path.dirname(os.path.abspath(__file__)) / Path("../images")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(100):
        image_file = image_dir / "%05d.png" % (i+3)
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        assert image.shape == (3, 360, 540)

        # for v in image.reshape(-1):
        #     print(struct.pack('B', v))
        #     print(v)
        #     break

        with open(save_dir / "%05d" % (i+3), 'wb') as f:
            for v in image.reshape(-1):
                f.write(struct.pack('B', v))

def convert_7scenes():
    scene_folder = Path('/home/nhsmt1123/master-thesis/dvmvs-downsample2/7scenes')
    test_dataset_names = ["chess-seq-01", "chess-seq-02", "fire-seq-01", "fire-seq-02", "heads-seq-02", "office-seq-01", "office-seq-03", "redkitchen-seq-01", "redkitchen-seq-07"]

    for test_dataset_name in test_dataset_names:
        save_dir = os.path.dirname(os.path.abspath(__file__)) / Path("../images") / test_dataset_name
        os.makedirs(save_dir, exist_ok=True)

        shutil.copy(scene_folder / test_dataset_name / "K.txt", save_dir)
        shutil.copy(scene_folder / test_dataset_name / "poses.txt", save_dir)

        image_filenames = sorted((scene_folder / test_dataset_name / 'images').files("*.png"))

        for image_file in image_filenames:
            image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            assert image.shape == (3, 256, 320)

            filename = image_file.split('/')[-1][:-4]

            with open(save_dir / filename, 'wb') as f:
                for v in image.reshape(-1):
                    f.write(struct.pack('B', v))



if __name__ == '__main__':
    # convert_hololens()
    convert_7scenes()
