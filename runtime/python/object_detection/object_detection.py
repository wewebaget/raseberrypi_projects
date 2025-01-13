#!/usr/bin/env python3
from dimits import Dimits
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from loguru import logger
import queue
import threading
from PIL import Image
from typing import List
from object_detection_utils import ObjectDetectionUtils
import subprocess
import serial
import time
import datetime
from collections import namedtuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference, load_input_images, validate_images, divide_list_to_batches

# Global debug flag
DEBUG = True

# Decorator to print function names
def debug_function(func):
    def wrapper(*args, **kwargs):
        if DEBUG:
            print(f"Calling function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Setup
who1 = "Justin"
who2 = "Zoe"

# Dictionary setup
w_t_bring = {
    ("sunday", who1): {"binder"},
    ("tuesday", who1): {"book"},
    ("monday", who1): {"person", "clock"},
    ("sunday", who2): {"teddy bear"},
    ("tuesday", who2): {"Pens", "Binder"},
    ("monday", who2): {"Bottle"}
}

# UID to who dictionary
who_i_i = {
    "f7 4b db 85": who1,
    "07 05 20 86": who2
}

# UID to objects mapping
uid_to_objects = {
    "2e a5 03 02": ["bottle", "person"],
    "67 c8 7e 00": ["binder"],
    # Add more mappings as needed
}

# Arduino setup
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)

# Initialize Dimits with the desired voice model
zoe = Dimits("en_US-amy-low")
mom = Dimits("en_US-kathleen-low")
justin = Dimits("en_US-danny-low")

def read(sorp : bool):

    if sorp == True:
        while True:
            uid = arduino.readline().decode("utf-8").strip()
            if uid != "":
                break
    else :
        start = time.time()
        passed = time.time() - start
        while True:
            uid = arduino.readline().decode("utf-8").strip()
            if uid != "" or passed >= 3:
                break
            passed = time.time() - start


    return uid

def recognize(uid):
    if uid not in who_i_i:
        logger.error(f"unrecognized")
        print(uid)
        valid = False
        who = None
        return who, valid
    else:
        print(who_i_i[uid])
        valid = True
        return who_i_i[uid], valid

def pic():
    command = "rm -f /home/Zoe/Hailo-Application-Code-Examples/runtime/python/object_detection/bath_images/*; rpicam-still -o /home/Zoe/Hailo-Application-Code-Examples/runtime/python/object_detection/bath_images/ --timelapse 100 --timestamp=1 --datetime=1 -t 3000"
    subprocess.run(command, shell=True)
    print("pic done")

def enqueue_images(images: List[Image.Image], batch_size: int, input_queue: queue.Queue, width: int, height: int, utils: ObjectDetectionUtils) -> None:
    for batch in divide_list_to_batches(images, batch_size):
        processed_batch = []
        batch_array = []
        for image in batch:
            processed_image = utils.preprocess(image, width, height)
            processed_batch.append(processed_image)
            batch_array.append(np.array(processed_image))

        input_queue.put(processed_batch)

    input_queue.put(None)

def process_output(output_queue: queue.Queue, output_path: Path, width: int, height: int, utils: ObjectDetectionUtils, detected_objects_list: list) -> None:
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break

        processed_image, infer_results = result
        detections = utils.extract_detections(infer_results)

        if len(infer_results) == 1:
            infer_results = infer_results[0]

        detections = utils.extract_detections(infer_results)

        for item_id in range(len(detections["detection_classes"])):
            confidence = detections["detection_scores"][item_id]

            if confidence > 0/5:
                label = utils.labels[detections["detection_classes"][item_id]]
                detected_objects_list.append(label)
        
        utils.visualize(detections, processed_image, image_id, output_path, width, height)
        image_id += 1

    output_queue.task_done()

def infer(images: List[Image.Image], net_path: str, labels_path: str, batch_size: int, output_path: Path) -> list:
    utils = ObjectDetectionUtils(labels_path)

    input_queue = queue.Queue()
    output_queue = queue.Queue()

    hailo_inference = HailoAsyncInference(net_path, input_queue, output_queue, batch_size)
    height, width, _ = hailo_inference.get_input_shape()

    enqueue_thread = threading.Thread(target=enqueue_images, args=(images, batch_size, input_queue, width, height, utils))
    detected = []
    process_thread = threading.Thread(target=process_output, args=(output_queue, output_path, width, height, utils, detected))

    enqueue_thread.start()
    process_thread.start()

    hailo_inference.run()

    enqueue_thread.join()
    output_queue.put(None)
    process_thread.join()

    logger.info(f'Inference was successful! Results have been saved in {output_path}')
    return detected

def object_detection() -> list:
    Args = namedtuple('Args', ['input', 'net', 'labels'])
    args = Args("/home/Zoe/Hailo-Application-Code-Examples/runtime/python/object_detection/bath_images/", 
    '/usr/share/hailo-models/yolov6n_h8l.hef', 
    '/home/Zoe/Hailo-Application-Code-Examples/runtime/python/object_detection/coco.txt')

    images = load_input_images(args.input)
    batch =  len(images)
    try:
        validate_images(images, batch)
    except ValueError as e:
        logger.error(e)
        return

    output_path = Path('output_images')
    output_path.mkdir(exist_ok=True)

    return infer(images, args.net, args.labels, batch, output_path)

def what_t_b(when: str, who: str) -> list:
    return w_t_bring[when, who]

def missing(target, have):
    missing = []
    for x in target:
        found = False
        if not (have is None):
            for y in have:
                if(x == y):
                    found = True
                    break
        if not found:
            missing.append(x)
    return missing

def tell(today, who, miss):
    if len(miss) > 0:
        text = "You forget "
        for i in range(len(miss)):
            if(i == len(miss)-1):
                if (len(miss) > 1):
                    text = text + "and " + miss[i]
                else:
                    text = text + miss[i]
                text = text + "."
            else:
                text = text + miss[i] + ", "
    else:
        text = "You are good to go!"
    text = text +" Today's "+ today+". " + who
    print(text)
    zoe.text_2_speech(text,  engine="aplay")

@debug_function
def recognize_user() -> str:
    while True:
        uid = read(True)
        who, valid = recognize(uid)
        if valid:
            return who

@debug_function
def rfid_scan(scan_duration: int = 3) -> list:
    detected_objects = set()

    start_time = time.time()

    passed = 0
    while passed < scan_duration:
        uid = read(False)

        if uid in uid_to_objects:
            detected_objects.update(uid_to_objects[uid])

        passed = time.time() - start_time

    return list(detected_objects)

@debug_function
def capture_and_detect_objects() -> list:
    pic()
    detected_from_camera = object_detection()
    detected_from_rfid = rfid_scan()
    combined_detected_objects = list(set(detected_from_camera + detected_from_rfid))
    return combined_detected_objects

@debug_function
def get_required_items(who: str) -> list:
    today = datetime.datetime.now().strftime("%A").lower()
    return what_t_b(today, who)

@debug_function
def find_missing_items(required_items: list, detected_items: list) -> list:
    return [item for item in required_items if item not in detected_items]

@debug_function
def notify_user(who: str, missing_items: list) -> None:
    today = datetime.datetime.now().strftime("%A").lower()
    tell(today, who, missing_items)

@debug_function
def main() -> None:
    print("done")
    while True:
        who = recognize_user()
        detected_objects = capture_and_detect_objects()
        required_items = get_required_items(who)
        missing_items = find_missing_items(required_items, detected_objects)
        notify_user(who, missing_items)

if __name__ == "__main__":
    main()