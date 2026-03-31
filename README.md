# ArUco Marker Generator

This project provides a simple Python script to generate ArUco markers for computer vision and robotics tasks.

## Installation

1.  Install the required dependencies:
    ```bash
    py -m pip install -r requirements.txt
    ```

## Usage

Run the script to generate markers. On Windows, it is recommended to use the `py` launcher:

```bash
py generate_aruco.py
```

### Options

You can customize the generation using command-line arguments:

-   `--num`: Number of markers to generate (default: 10).
-   `--size`: Size of each marker in pixels (default: 1000).
-   `--dict`: The ArUco dictionary to use (default: `DICT_4X4_50`).
-   `--dir`: Output directory for the images (default: `markers`).

Example:
```bash
python generate_aruco.py --num 20 --size 1000 --dict DICT_6X6_100 --dir my_markers
```

### Printing Tips

-   The generated markers include a **white border** to ensure they are easily detectable even when placed on dark surfaces.
-   When printing, ensure you do not use "Fit to Page" if you need specific physical dimensions. 
-   The default size (400px) is sufficient for most desktop printers and will result in a clear marker even at smaller physical sizes.
-   If using these for distance estimation, remember to measure the physical width of the marker (the black part) after printing.
