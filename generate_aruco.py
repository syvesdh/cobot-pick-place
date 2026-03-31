import cv2
import os
import argparse
import numpy as np

def generate_markers(dictionary_type=cv2.aruco.DICT_4X4_50, 
                     num_markers=10, 
                     marker_size=1000, 
                     output_dir="markers",
                     border_bits=1):
    """
    Generates a set of ArUco markers and saves them as PNG files.
    
    Args:
        dictionary_type: One of the cv2.aruco.DICT_* constants.
        num_markers: Number of markers to generate (starting from ID 0).
        marker_size: Size of the marker image in pixels (side).
        output_dir: Directory where the markers will be saved.
        border_bits: Thickness of the marker's border in bits.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Access the predefined dictionary
    # For newer OpenCV (4.7+), we use getPredefinedDictionary
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
    except AttributeError:
        # Fallback for older OpenCV naming if needed
        dictionary = cv2.aruco.Dictionary_get(dictionary_type)

    print(f"Generating {num_markers} markers from dictionary {dictionary_type}...")

    for i in range(num_markers):
        # Generate the marker image
        # Modern API: generateImageMarker
        try:
            marker_img = cv2.aruco.generateImageMarker(dictionary, i, marker_size, borderBits=border_bits)
        except AttributeError:
            # Fallback for older OpenCV: drawMarker
            marker_img = cv2.aruco.drawMarker(dictionary, i, marker_size, borderBits=border_bits)

        # To make it easier to cut out, we can add a white border around the black marker
        # Standard ArUco markers have a black bit border, but sometimes adding a 
        # physical white margin helps detection in messy environments.
        white_margin = 20
        marker_with_margin = cv2.copyMakeBorder(
            marker_img, 
            white_margin, white_margin, white_margin, white_margin, 
            cv2.BORDER_CONSTANT, value=255
        )

        filename = os.path.join(output_dir, f"marker_{i:03d}.png")
        cv2.imwrite(filename, marker_with_margin)
        
    print(f"Done! Markers saved in '{output_dir}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ArUco markers for printing.")
    parser.add_argument("--num", type=int, default=10, help="Number of markers to generate (default: 10)")
    parser.add_argument("--size", type=int, default=1000, help="Size of each marker in pixels (default: 1000)")
    parser.add_argument("--dict", type=str, default="DICT_4X4_50", 
                        help="Dictionary name (e.g., DICT_4X4_50, DICT_6X6_250)")
    parser.add_argument("--dir", type=str, default="markers", help="Output directory (default: markers)")

    args = parser.parse_args()

    # Map string dictionary name to cv2 constant
    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    }

    if args.dict not in dict_map:
        print(f"Error: Unknown dictionary '{args.dict}'. Choose from: {list(dict_map.keys())}")
    else:
        generate_markers(
            dictionary_type=dict_map[args.dict], 
            num_markers=args.num, 
            marker_size=args.size, 
            output_dir=args.dir
        )
