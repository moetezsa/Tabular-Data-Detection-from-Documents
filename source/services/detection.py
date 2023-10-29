from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
from source.datamodel.output import Output
import numpy as np
from typing import List


def detection(image: np.ndarray, image_size: List[int], image_name: str):
    processor = AutoFeatureExtractor.from_pretrained("moetezsa/table_detection_v1")
    model = AutoModelForObjectDetection.from_pretrained("moetezsa/table_detection_v1")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=[(image_size[0], image_size[1])])[
        0
    ]
    boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        result = Output(
                image_name=image_name,
                bbox=box
        )
        boxes.append(result)
    return boxes


