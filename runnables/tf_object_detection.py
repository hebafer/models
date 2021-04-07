import os
import argparse

import numpy as np
from PIL import Image

import tensorflow as tf
from object_detection.utils import label_map_util


def build_argument_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)

    return parser


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


if __name__ == "__main__":
    #parser = build_argument_parser()
    #args, _ = parser.parse_known_args()
    #columns = args.columns.split(',')
    #bin_method_array = args.bin_method.split(',') if args.bin_method else None
    #bin_num_array = [int(item) for item in args.bin_num.split(',')] if args.bin_num else None

    select_input = os.getenv("SQLFLOW_TO_RUN_SELECT")
    output = os.getenv("SQLFLOW_TO_RUN_INTO")
    #output_tables = output.split(',')
    datasource = os.getenv("SQLFLOW_DATASOURCE")

    tf.keras.backend.clear_session()
    print('Building model and restoring weights for fine-tuning...', flush=True)
    path_to_saved_model = 'object_detection/models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model/'
    path_to_labels = 'object_detection/labels/mscoco_label_map.pbtxt'

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(path_to_saved_model)
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

    test_image_dir = os.path.abspath('../datasets/voc_simple/test/JPEGImages/test/')

    # Create Test Dataset with 5 images
    test_images = []
    for i in range(0, 5):
        image_path = os.path.join(test_image_dir, str(i) + '.jpg')
        test_images.append(image_path)

    # Note that the first frame will trigger tracing of the tf.function, which will
    # take some time, after which inference should be fast.
    for image_path in test_images:

        print('Running inference for {}... '.format(image_path), end='')
        image_np = load_image_into_numpy_array(image_path)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        output_dict = detect_fn(input_tensor)

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0]
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        objects = []
        for index, value in enumerate(output_dict['detection_classes']):
            display_str_dict = {
                'name': category_index[value.numpy()]['name'],
                'score': '{}%'.format(output_dict['detection_scores'][index].numpy()),
            }
            objects.append(display_str_dict)
        print(objects)
