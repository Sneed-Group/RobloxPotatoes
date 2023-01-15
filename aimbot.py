import pyautogui
import win32api, win32con, win32gui
import cv2
import math
import time
import argparse
import os
import json
import numpy as np
from threading import Lock

# printing only warnings and error messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

try:
    import tensorflow as tf
    from PIL import Image
except ImportError:
    raise ImportError("ERROR: Failed to import libraries. Please refer to READEME.md file\n")

EXPORT_MODEL_VERSION = 1


class TFModel:
    def __init__(self, dir_path) -> None:
        # Assume model is in the parent directory for this file
        self.model_dir = os.path.dirname(dir_path)
        # make sure our exported SavedModel folder exists
        with open(os.path.join(self.model_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.model_file = os.path.join(self.model_dir, self.signature.get("filename"))
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"Model file does not exist")
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")
        self.lock = Lock()

        # loading the saved model
        self.model = tf.saved_model.load(tags=self.signature.get("tags"), export_dir=self.model_dir)
        self.predict_fn = self.model.signatures["serving_default"]

        # Look for the version in signature file.
        # If it's not found or the doesn't match expected, print a message
        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(
                f"There has been a change to the model format. Please use a model with a signature 'export_model_version' that matches {EXPORT_MODEL_VERSION}."
            )

    def predict(self, image: Image.Image) -> dict:
        with self.lock:
            # create the feed dictionary that is the input to the model
            feed_dict = {}
            # first, add our image to the dictionary (comes from our signature.json file)
            feed_dict[list(self.inputs.keys())[0]] = tf.convert_to_tensor(image)
            # run the model!
            outputs = self.predict_fn(**feed_dict)
            # return the processed output
            return self.process_output(outputs)

    def process_output(self, outputs) -> dict:
        # do a bit of postprocessing
        out_keys = ["label", "confidence"]
        results = {}
        # since we actually ran on a batch of size 1, index out the items from the returned numpy arrays
        for key, tf_val in outputs.items():
            val = tf_val.numpy().tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output

size_scale = 3
model = TFModel(dir_path=os.path.dirname(__file__))
while True:
    # Get rect of Window
    hwnd = win32gui.FindWindow(None, 'Roblox')
    #hwnd = win32gui.FindWindow("UnrealWindow", None) # Fortnite
    rect = win32gui.GetWindowRect(hwnd)
    region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]

    # Get image of screen
    ori_img = np.array(pyautogui.screenshot(region=region))
    ori_img = cv2.resize(ori_img, (ori_img.shape[1] // size_scale, ori_img.shape[0] // size_scale))
    image = np.expand_dims(ori_img, 0)
    img_w, img_h = image.shape[2], image.shape[1]

    # Detection
    outputs = model.predict(image)
    result = {key:value.numpy() for key,value in result.items()}
    boxes = result['detection_boxes'][0]
    scores = result['detection_scores'][0]
    classes = result['detection_classes'][0]

    # Check every detected object
    detected_boxes = []
    for i, box in enumerate(boxes):
        # Choose only person(class:1)
        if classes[i] == 1 and scores[i] >= 0.5:
            ymin, xmin, ymax, xmax = tuple(box)
            if ymin > 0.5 and ymax > 0.8: # CS:Go
            #if int(xmin * img_w * 3) < 450: # Fortnite
                continue
            left, right, top, bottom = int(xmin * img_w), int(xmax * img_w), int(ymin * img_h), int(ymax * img_h)
            detected_boxes.append((left, right, top, bottom))
            #cv2.rectangle(ori_img, (left, top), (right, bottom), (255, 255, 0), 2)

    print("Detected:", len(detected_boxes))

    # Check Closest
    if len(detected_boxes) >= 1:
        min = 99999
        at = 0
        centers = []
        for i, box in enumerate(detected_boxes):
            x1, x2, y1, y2 = box
            c_x = ((x2 - x1) / 2) + x1
            c_y = ((y2 - y1) / 2) + y1
            centers.append((c_x, c_y))
            dist = math.sqrt(math.pow(img_w/2 - c_x, 2) + math.pow(img_h/2 - c_y, 2))
            if dist < min:
                min = dist
                at = i

        # Pixel difference between crosshair(center) and the closest object
        x = centers[at][0] - img_w/2
        y = centers[at][1] - img_h/2 - (detected_boxes[at][3] - detected_boxes[at][2]) * 0.45

        # Move mouse and shoot
        scale = 1.7 * size_scale
        x = int(x * scale)
        y = int(y * scale)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

    time.sleep(0.1)