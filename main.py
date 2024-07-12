import cv2
from object_detection.detector import ObjectDetection
from object_detection.sliding_window_attention import SlidingWindowAttention

def main():
    image_path = 'sample.png'
    model_url = 'https://tfhub.dev/tensorflow/efficientdet/d0/1'
    
    window_size = (300, 300)  # Adjust window size as needed
    step_size = (150, 150)    # Adjust step size as needed

    attention_module = SlidingWindowAttention(window_size, step_size)
    detector = ObjectDetection(model_url, attention_module)

    image = cv2.imread(image_path)
    detections = detector.detect_objects(image)

    print(f"Total detections: {len(detections)}")

    for confidence, box in detections:
        print(f"Detected object with confidence {confidence:.2f}")
        y_min, x_min, y_max, x_max = box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(image, f"Confidence: {confidence:.2f}", (int(x_min), int(y_min) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
