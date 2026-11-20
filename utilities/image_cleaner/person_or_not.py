"""
Batch process images to identify those with detectable human poses.

Uses MoveNet MultiPose Lightning, the same model the web page runs on the
visitor's webcam (via ml5.js), and prepares each image the same way the page
does. That matters: the stored pose for an artwork should be what the page's
own model would see in it. (An earlier version used MoveNet SinglePose here;
the two models disagree often enough that an artwork was the best match for
its own pose only about half the time.)

Outputs:
- /processed_images/ folder with filtered images
- image_metadata.json with metadata + L2 vectors for matching
"""

import os
import json
import math
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# ============== CONFIGURATION ==============

# Paths
INPUT_IMAGES_FOLDER = "input_images"
OUTPUT_IMAGES_FOLDER = "processed_images"
INPUT_METADATA_FILE = "input_metadata.json"
OUTPUT_METADATA_FILE = "image_metadata.json"

# Output image sizing: downscale to this max dimension (long edge) and
# recompress. The display canvas is 400x593 and the info page shows up to
# 800px wide, so 1200px covers retina displays while keeping files ~100-200KB
# instead of multi-MB museum masters.
MAX_OUTPUT_DIMENSION = 1200
OUTPUT_JPEG_QUALITY = 82

# Pose detection thresholds
# Minimum average confidence across all keypoints to consider a pose "detected"
MIN_AVERAGE_CONFIDENCE = 0.3
# Minimum number of keypoints that must exceed confidence threshold
MIN_CONFIDENT_KEYPOINTS = 10
# Confidence threshold for individual keypoints
KEYPOINT_CONFIDENCE_THRESHOLD = 0.1

# MoveNet model URL (MultiPose Lightning v1 - the model sketch.js loads from
# libs/movenet-multipose-lightning-1/)
MOVENET_MODEL_URL = "https://tfhub.dev/google/movenet/multipose/lightning/1"
# The web page's pose library (ml5.js / TensorFlow.js pose-detection) shrinks
# each frame so its longer side is this many pixels, and ignores any detected
# person scoring below MIN_POSE_SCORE. Both are that library's defaults; keep
# them in step with it so this script sees what the page sees.
MODEL_MAX_DIMENSION = 256
MIN_POSE_SCORE = 0.25

# MoveNet keypoint names (same order as ml5.js)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


# ============== NORMALIZATION FUNCTIONS ==============
# These match the JavaScript functions in sketch.js exactly

def compute_bounding_box(keypoints, confidence_threshold=KEYPOINT_CONFIDENCE_THRESHOLD):
    """Compute bounding box from keypoints with confidence threshold."""
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    for kp in keypoints:
        if kp['confidence'] > confidence_threshold:
            x_min = min(x_min, kp['x'])
            x_max = max(x_max, kp['x'])
            y_min = min(y_min, kp['y'])
            y_max = max(y_max, kp['y'])

    return {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}


def normalize_keypoints(keypoints, confidence_threshold=KEYPOINT_CONFIDENCE_THRESHOLD):
    """Normalize keypoints: center around bounding box and scale to unit square."""
    box = compute_bounding_box(keypoints, confidence_threshold)

    # Compute center and size of bounding box
    center_x = (box['x_min'] + box['x_max']) / 2
    center_y = (box['y_min'] + box['y_max']) / 2
    box_width = box['x_max'] - box['x_min']
    box_height = box['y_max'] - box['y_min']

    # Use the larger dimension to maintain aspect ratio
    scale = max(box_width, box_height)

    # Avoid division by zero
    if scale == 0:
        scale = 1

    # Normalize each keypoint
    normalized = []
    for kp in keypoints:
        normalized.append({
            'name': kp['name'],
            'x': (kp['x'] - center_x) / scale,
            'y': (kp['y'] - center_y) / scale,
            'confidence': kp['confidence']
        })

    return normalized


def keypoints_to_vector(keypoints):
    """Flatten keypoints to a vector [x1, y1, x2, y2, ...] for L2 normalization."""
    vector = []
    for kp in keypoints:
        vector.append(kp['x'])
        vector.append(kp['y'])
    return vector


def l2_normalize(vector):
    """L2 normalize a vector (divide by magnitude to create unit vector)."""
    sum_of_squares = sum(v * v for v in vector)
    magnitude = sum_of_squares ** 0.5

    # Avoid division by zero
    if magnitude == 0:
        return vector

    return [v / magnitude for v in vector]


def process_pose(keypoints):
    """Process keypoints: normalize and apply L2 normalization."""
    normalized_keypoints = normalize_keypoints(keypoints)
    vector = keypoints_to_vector(normalized_keypoints)
    l2_vector = l2_normalize(vector)

    return {
        'normalized_keypoints': normalized_keypoints,
        'l2_vector': l2_vector
    }


# ============== POSE DETECTION FUNCTIONS ==============

def load_model():
    """Load MoveNet model from TensorFlow Hub."""
    print("Loading MoveNet model...")
    model = hub.load(MOVENET_MODEL_URL)
    movenet = model.signatures['serving_default']
    print("Model loaded successfully.")
    return movenet


def load_and_preprocess_image(image_path):
    """Load an image and prepare it for MoveNet MultiPose exactly as the web
    page's pose library does: shrink it so the longer side is
    MODEL_MAX_DIMENSION, then pad the bottom or right edge with black up to
    the next multiple of 32 (the model needs both sides to be multiples of 32).

    Returns the model input and the sizes needed to map keypoints back to
    the original image.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    original_height = int(tf.shape(image)[0])
    original_width = int(tf.shape(image)[1])

    # int(x + 0.5) rounds halves up, like JavaScript's Math.round
    if original_width > original_height:
        resized_width = MODEL_MAX_DIMENSION
        resized_height = max(1, int(MODEL_MAX_DIMENSION * original_height / original_width + 0.5))
    else:
        resized_height = MODEL_MAX_DIMENSION
        resized_width = max(1, int(MODEL_MAX_DIMENSION * original_width / original_height + 0.5))
    padded_height = 32 * math.ceil(resized_height / 32)
    padded_width = 32 * math.ceil(resized_width / 32)

    # Plain bilinear resize with the same pixel alignment TensorFlow.js uses
    resized = tf.compat.v1.image.resize_bilinear(
        tf.expand_dims(tf.cast(image, tf.float32), axis=0),
        [resized_height, resized_width],
        align_corners=False, half_pixel_centers=False)
    padded = tf.pad(resized, [[0, 0],
                              [0, padded_height - resized_height],
                              [0, padded_width - resized_width],
                              [0, 0]])
    input_image = tf.cast(padded, dtype=tf.int32)

    sizes = {
        'original_width': original_width, 'original_height': original_height,
        'resized_width': resized_width, 'resized_height': resized_height,
        'padded_width': padded_width, 'padded_height': padded_height,
    }
    return input_image, sizes


def detect_pose(movenet, image_path):
    """Run pose detection on an image and return the keypoints of the person
    the web page would match against, or None if it finds nobody.

    MoveNet MultiPose reports up to 6 people. Like the page, this ignores
    anyone scoring below MIN_POSE_SCORE and then takes whoever occupies the
    most space (largest bounding box around their confident keypoints).
    """
    input_image, sizes = load_and_preprocess_image(image_path)

    # Output shape [1, 6, 56]: per person, 17 keypoints as (y, x, confidence),
    # then a bounding box (4 numbers) and an overall score
    people = movenet(input_image)['output_0'].numpy()[0]

    # Keypoint coordinates come back as fractions (0-1) of the PADDED input.
    # Convert to pixels of the padded input, then scale up to the original
    # image. Both axes use the same scale factor apart from rounding, so the
    # pose keeps its true proportions whatever the image's shape.
    x_scale = sizes['padded_width'] * sizes['original_width'] / sizes['resized_width']
    y_scale = sizes['padded_height'] * sizes['original_height'] / sizes['resized_height']

    best_keypoints = None
    best_area = -1
    for person in people:
        if person[55] < MIN_POSE_SCORE:
            continue
        keypoints = []
        for i, name in enumerate(KEYPOINT_NAMES):
            y, x, confidence = person[i * 3: i * 3 + 3]
            keypoints.append({
                'name': name,
                'x': float(x * x_scale),   # original image pixels
                'y': float(y * y_scale),
                'confidence': float(confidence)
            })
        box = compute_bounding_box(keypoints)
        if box['x_min'] == float('inf'):
            continue  # no confident keypoints at all
        area = (box['x_max'] - box['x_min']) * (box['y_max'] - box['y_min'])
        if area > best_area:
            best_area = area
            best_keypoints = keypoints

    return best_keypoints


def save_resized_image(source_path, output_path):
    """Save a web-sized copy of the image: downscale to MAX_OUTPUT_DIMENSION
    on the long edge and recompress as progressive JPEG."""
    from PIL import ImageOps
    with Image.open(source_path) as im:
        im = ImageOps.exif_transpose(im)
        im.thumbnail((MAX_OUTPUT_DIMENSION, MAX_OUTPUT_DIMENSION), Image.LANCZOS)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(output_path, "JPEG", quality=OUTPUT_JPEG_QUALITY,
                optimize=True, progressive=True)


def is_valid_pose(keypoints):
    """Check if the detected pose meets our quality thresholds."""
    if keypoints is None:
        return False

    # Count keypoints above confidence threshold
    confident_keypoints = sum(
        1 for kp in keypoints
        if kp['confidence'] > KEYPOINT_CONFIDENCE_THRESHOLD
    )

    # Calculate average confidence
    avg_confidence = sum(kp['confidence'] for kp in keypoints) / len(keypoints)

    # Check both criteria
    meets_count = confident_keypoints >= MIN_CONFIDENT_KEYPOINTS
    meets_avg = avg_confidence >= MIN_AVERAGE_CONFIDENCE

    return meets_count and meets_avg


# ============== MAIN PROCESSING FUNCTION ==============

def load_metadata():
    """Load the Met metadata JSON file."""
    print(f"Loading metadata from {INPUT_METADATA_FILE}...")
    with open(INPUT_METADATA_FILE, 'r') as f:
        metadata_list = json.load(f)

    # Convert to dictionary keyed by Object_ID for easy lookup
    metadata_dict = {}
    for item in metadata_list:
        object_id = str(item.get('Object_ID', ''))
        if object_id:
            metadata_dict[object_id] = item

    print(f"Loaded metadata for {len(metadata_dict)} objects.")
    return metadata_dict


def process_all_images():
    """Main function to process all images and filter those with valid poses."""

    # Load the MoveNet model
    movenet = load_model()

    # Load metadata
    metadata_dict = load_metadata()

    # Create output folder if it doesn't exist
    if not os.path.exists(OUTPUT_IMAGES_FOLDER):
        os.makedirs(OUTPUT_IMAGES_FOLDER)
        print(f"Created output folder: {OUTPUT_IMAGES_FOLDER}")

    # Get list of image files
    image_files = [f for f in os.listdir(INPUT_IMAGES_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    print(f"Found {total_images} images to process.")

    # Track results
    processed_count = 0
    valid_pose_count = 0
    failed_count = 0
    output_metadata = []

    # Process each image
    for i, filename in enumerate(image_files):
        # Progress update every 50 images
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing image {i + 1}/{total_images}...")

        image_path = os.path.join(INPUT_IMAGES_FOLDER, filename)

        # Extract Object_ID from filename (assumes format: object_id.jpg)
        object_id = os.path.splitext(filename)[0]

        # Detect pose (None = no person found)
        try:
            keypoints = detect_pose(movenet, image_path)
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            failed_count += 1
            continue
        processed_count += 1

        # Check if valid pose
        if is_valid_pose(keypoints):
            valid_pose_count += 1

            # Process pose to get L2 vector
            pose_data = process_pose(keypoints)

            # Save a web-sized copy to the output folder. The copy is always
            # JPEG data, so it always gets a .jpg name — a PNG input keeps its
            # object_id but must not keep its .png extension
            output_filename = object_id + ".jpg"
            output_path = os.path.join(OUTPUT_IMAGES_FOLDER, output_filename)
            save_resized_image(image_path, output_path)

            # Get metadata for this image
            image_metadata = metadata_dict.get(object_id, {})

            # Create output record with metadata and pose data
            output_record = {
                'object_id': object_id,
                'filename': output_filename,
                'l2_vector': pose_data['l2_vector'],
                # Per-keypoint confidences (same order as KEYPOINT_NAMES) so
                # the matcher can ignore vector dimensions where the artwork's
                # own keypoints were uncertain
                'keypoint_confidences': [round(kp['confidence'], 4) for kp in keypoints],
                'metadata': {
                    'title': image_metadata.get('Title', ''),
                    'artist': image_metadata.get('Artist_Display_Name', ''),
                    'date': image_metadata.get('Object_Date', ''),
                    'department': image_metadata.get('Department', ''),
                    'medium': image_metadata.get('Medium', ''),
                    'link': image_metadata.get('Link_Resource', ''),
                    'repository': image_metadata.get('Repository', '')
                }
            }
            output_metadata.append(output_record)


    # Save output metadata
    print(f"\nSaving metadata to {OUTPUT_METADATA_FILE}...")
    with open(OUTPUT_METADATA_FILE, 'w') as f:
        json.dump(output_metadata, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total images processed: {processed_count}")
    print(f"Images with valid poses: {valid_pose_count}")
    print(f"Images without valid poses: {processed_count - valid_pose_count}")
    print(f"Images that could not be read: {failed_count}")
    print(f"\nOutput saved to:")
    print(f"  - Images: {OUTPUT_IMAGES_FOLDER}/")
    print(f"  - Metadata: {OUTPUT_METADATA_FILE}")


# ============== ENTRY POINT ==============

#using 'if __name__...' allows you to import this script as a library and call functions within it
#so, for example, you would 'import person_or_not' and then call 'person_or_not.is_valid_pose()
if __name__ == "__main__":
    process_all_images()