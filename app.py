import tensorflow as tf
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import streamlit as st
import cv2 
import numpy as np
import os
from matplotlib import pyplot as plt

"""
# Proyecto ToM: Tomatoes Monitor
"""
st.image("https://github.com/rrequeena/ToM/blob/main/ToM-logo2.jpg?raw=true")
"""
### WebApp base del proyecto ToM (Fase de Pruebas).
Suba una imagen que contenga tomates para probar la detección de los mismos
"""

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file('tomato_detection/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('tomato_detection', 'ckpt-3')).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(
	'tomato_detection/label_map.pbtxt')

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

uploaded_image = st.file_uploader("Subir una imagen...")

if uploaded_image is not None:

	img_uploaded_data = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
	opencv_image = cv2.imdecode(img_uploaded_data, 1)
	img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
	print("\n\n==============\n\n")
	print(opencv_image.shape)
	print("\n\n==============\n\n")
	image_np = np.array(img)

	input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
	detections = detect_fn(input_tensor)

	num_detections = int(detections.pop('num_detections'))
	detections = {key: value[0, :num_detections].numpy()
	              for key, value in detections.items()}
	detections['num_detections'] = num_detections
	
	# detection_classes should be ints.
	detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

	label_id_offset = 1
	image_np_with_detections = image_np.copy()

	viz_utils.visualize_boxes_and_labels_on_image_array(
	            image_np_with_detections,
	            detections['detection_boxes'],
	            detections['detection_classes']+label_id_offset,
	            detections['detection_scores'],
	            category_index,
	            use_normalized_coordinates=True,
	            max_boxes_to_draw=5,
	            min_score_thresh=.8,
	            agnostic_mode=False,
	            line_thickness=8)
	"""
	## Detecciones
	"""
	st.image(image_np_with_detections)
