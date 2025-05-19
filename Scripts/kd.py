import darknet
import numpy as np
import cv2
import os
import random
import time
import ctypes
from ctypes import c_char_p

# Configuration parameters
TEACHER_CONFIG = "mycfg/yolov4-tiny_helmet.cfg"
TEACHER_WEIGHTS = "mybackup/yolov4-tiny_helmet_final.weights"
STUDENT_CONFIG = "mycfg/yolofv1_helmetv2_reduce_filter_gray.cfg"
STUDENT_WEIGHTS = "mybackup/yolofv1_helmetv2_reduce_filter_gray_best.weights"
DATA_CONFIG = "helmet_datasetv2_1/helmetv2.data"
CLASS_NAMES = "helmet_datasetv2_1/helmetv2.names"
BATCH_SIZE = 16
MAX_ITERATIONS = 10000
TEMPERATURE = 2.0

def load_models():
    """Load the teacher and student models"""
    teacher_net = darknet.load_net_custom(
        TEACHER_CONFIG.encode("ascii"), 
        TEACHER_WEIGHTS.encode("ascii"), 
        0, 
        1
    )
    student_net = darknet.load_net_custom(
        STUDENT_CONFIG.encode("ascii"), 
        STUDENT_WEIGHTS.encode("ascii"), 
        0, 
        1
    )
    return teacher_net, student_net

def load_class_names(names_file):
    with open(names_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def get_detections(net, image_path, class_names, is_grayscale=False):
    """
    Process an image with the network and return the detections.
    Supports grayscale input for student models.
    """
    try:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None

        # Get network dimensions
        net_width = darknet.network_width(net)
        net_height = darknet.network_height(net)

        # Prepare image based on channel config
        if is_grayscale:
            # Convert grayscale to 3 channels (fake RGB) for Darknet
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_resized = cv2.resize(image_rgb, (net_width, net_height), interpolation=cv2.INTER_LINEAR)

        # Prepare darknet image
        darknet_image = darknet.make_image(net_width, net_height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

        detections = darknet.detect_image(net, class_names, darknet_image, 0.5, 0.5, 0.5)
        darknet.free_image(darknet_image)

        return detections
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def extract_detection_features(detections):
    """
    Extract features from detections to use for knowledge distillation
    Returns confidence scores, class probabilities and bounding box info
    """
    if not detections:
        return None
        
    confidences = []
    class_probs = []
    boxes = []
    
    for detection in detections:
        # Each detection is (label, confidence, (x, y, w, h))
        label, confidence, bbox = detection
        confidences.append(confidence)
        
        # For class probabilities, we'd ideally get this from an earlier layer
        # but for now we'll just use the confidence as a proxy
        class_probs.append(confidence)
        
        # Normalize bbox coordinates
        boxes.extend([bbox[0], bbox[1], bbox[2], bbox[3]])
    
    # Combine features
    features = np.array(confidences + class_probs + boxes, dtype=np.float32)
    return features

def compute_distillation_loss(teacher_features, student_features, temperature):
    """
    Compute knowledge distillation loss between teacher and student feature vectors
    """
    if teacher_features is None or student_features is None:
        return 0.0

    if not isinstance(teacher_features, np.ndarray) or not isinstance(student_features, np.ndarray):
        return 0.0

    if teacher_features.size == 0 or student_features.size == 0:
        return 0.0

        
    # Handle different lengths by padding the shorter one
    if len(teacher_features) != len(student_features):
        max_len = max(len(teacher_features), len(student_features))
        if len(teacher_features) < max_len:
            padding = np.zeros(max_len - len(teacher_features))
            teacher_features = np.concatenate([teacher_features, padding])
        else:
            padding = np.zeros(max_len - len(student_features))
            student_features = np.concatenate([student_features, padding])
    
    # Apply temperature scaling
    teacher_soft = softmax_stable(teacher_features, temperature)
    student_soft = softmax_stable(student_features, temperature)
    
    # Compute KL divergence loss
    epsilon = 1e-7
    loss = np.sum(teacher_soft * np.log((teacher_soft + epsilon) / (student_soft + epsilon)))
    
    return loss

def get_batch_from_dataset(data_path, batch_size):
    """Get a batch of image paths from the dataset"""
    with open(data_path, 'r') as f:
        image_paths = [line.strip() for line in f]
    
    selected_paths = random.sample(image_paths, min(batch_size, len(image_paths)))
    return selected_paths

def softmax_stable(logits, temperature):
    logits = logits / temperature
    logits -= np.max(logits)  # for numerical stability
    exp_logits = np.exp(logits)
    softmax_output = exp_logits / (np.sum(exp_logits) + 1e-7)
    return softmax_output

def train_distillation():
    """Main training loop for knowledge distillation monitoring"""
    print("Loading models...")
    teacher_net, student_net = load_models()
    print("Loading training data...")
    # Load class names
    class_names = load_class_names(CLASS_NAMES)

    # Extract train.txt path from DATA_CONFIG
    train_path = ""
    with open(DATA_CONFIG, 'r') as f:
        for line in f:
            if line.startswith("train"):
                train_path = line.split("=")[1].strip()
                break
    
    if not train_path:
        raise ValueError(f"Could not find train path in {DATA_CONFIG}")
    
    print(f"Training with data from {train_path}")
    
    # Training loop
    for iteration in range(MAX_ITERATIONS):
        start_time = time.time()
        
        # Get batch of training images
        batch_paths = get_batch_from_dataset(train_path, BATCH_SIZE)
        
        total_loss = 0
        valid_samples = 0
        
        for img_path in batch_paths:
            # Get detections from teacher and student
            teacher_detections = get_detections(teacher_net, img_path, class_names, is_grayscale=False)
            student_detections = get_detections(student_net, img_path, class_names, is_grayscale=True)
            
            # Extract features from detections
            teacher_features = extract_detection_features(teacher_detections)
            student_features = extract_detection_features(student_detections)
            
            if teacher_features is not None and student_features is not None:
                # Compute distillation loss
                loss = compute_distillation_loss(teacher_features, student_features, TEMPERATURE)
                total_loss += loss
                valid_samples += 1
        
        # Calculate average loss
        avg_loss = total_loss / valid_samples if valid_samples > 0 else 0
        elapsed = time.time() - start_time
        
        # Print progress
        print(f"Iteration {iteration+1}/{MAX_ITERATIONS}, "
              f"Avg Loss: {avg_loss:.6f}, "
              f"Valid Samples: {valid_samples}/{len(batch_paths)}, "
              f"Time: {elapsed:.2f}s")
        
        # Save checkpoint periodically
        if (iteration + 1) % 2 == 0:
            checkpoint_file = f"out/student_kd_checkpoint_{iteration+1}.weights"
            # Fix: Since student_net is an integer handle, pass it directly
            darknet.save_weights(student_net, checkpoint_file.encode("ascii"))
    
    # Save final model
    print("Saving final distilled model...")
    # Fix: Pass the handle directly to save_weights
    darknet.save_weights(student_net, "student_distilled_final.weights".encode("ascii"))
    print("Knowledge distillation monitoring complete!")

if __name__ == "__main__":
    train_distillation()