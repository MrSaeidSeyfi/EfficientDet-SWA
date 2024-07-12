import cv2
import numpy as np
import tensorflow_hub as hub
from .sliding_window_attention import SlidingWindowAttention
from tensorflow.image import non_max_suppression

class ObjectDetection:
    def __init__(self, model_url, attention_module):
        self.model = hub.load(model_url)
        self.attention_module = attention_module

    def preprocess_image(self, image, target_size):
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)
        return image

    def detect_objects(self, image):
        windows = self.attention_module.generate_windows(image)
        print(f"Generated {len(windows)} windows from the image")

        results = []
        for idx, (window, x_offset, y_offset) in enumerate(windows):
            print(f"Processing window {idx+1}/{len(windows)} at position ({x_offset}, {y_offset})")
            preprocessed_window = self.preprocess_image(window, (512, 512))
            detector_output = self.model(preprocessed_window)
            
            detection_boxes = detector_output['detection_boxes'][0].numpy()
            detection_scores = detector_output['detection_scores'][0].numpy()
            detection_classes = detector_output['detection_classes'][0].numpy()

            print(f"Detections in window {idx+1}: {len(detection_boxes)}")
            for i in range(len(detection_boxes)):
                score = detection_scores[i]
                if score > 0.4: 
                    box = detection_boxes[i]
                    box[0] = box[0] * window.shape[0] + y_offset
                    box[1] = box[1] * window.shape[1] + x_offset
                    box[2] = box[2] * window.shape[0] + y_offset
                    box[3] = box[3] * window.shape[1] + x_offset
                    results.append((score, box))
                else:
                    print(f"Low confidence detection: score={score}")
 
        if results:
            boxes = np.array([r[1] for r in results])
            scores = np.array([r[0] for r in results])
            boxes = boxes.reshape(-1, 4)  # Ensure boxes is 2-dimensional
            indices = non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=0.5)
            filtered_results = [(scores[i], boxes[i]) for i in indices.numpy()]
        else:
            filtered_results = []

        return filtered_results
