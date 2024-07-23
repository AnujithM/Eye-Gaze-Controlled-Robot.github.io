# Grid-based dynamic planner using monocular camera vision.

Path planning algorithms and upper limb segmentation using instance segmentation models are utilized to generate obstacle-free trajectories to the printing locations, mitigating interference from hand and forearm.
This mainly uses a monocular camera to create a grid-based environment to create trajectories for the robot to follow.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Requests
- Supervision (sv)
- Inference models (from `inference.models.utils`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AnujithM/Eye-Gaze-Controlled-Robot.github.io.git
    cd Eye-Gaze-Controlled-Robot.github.io/Planner
    ```

2. Install the required Python packages:
    ```sh
    pip install opencv-python-headless numpy requests supervision
    ```

## Usage

1. Replace `model_id` and `api_key` in the script with your specific model ID and API key.

2. Update the `url` variable with the correct MJPEG stream URL.

3. Run the script:
    ```sh
    python DynamicAstar_V12.py

## How It Works

1. **Fetch and Decode Frames:** The script continuously fetches frames from the specified MJPEG stream URL and decodes them using OpenCV.

2. **Hand Segmentation:** The frames are passed through a hand segmentation model to detect and segment hands in the frame.

3. **Grid and Adjacency List:** A grid overlay is created on the frame, dividing it into cells. An adjacency list is maintained to represent the connections between the grid cells.

4. **Dynamic A* Pathfinding:** The dynamic A* algorithm finds a path from a source cell to a goal cell, avoiding cells containing hands (red centroids).

5. **Frame Annotation:** The segmented frame is annotated with the detected hands, grid cells, and the path found by the A* algorithm.

6. **Real-time Display:** The annotated frame is displayed in real-time, showing the segmented hands, grid, and path.

## Results

![](Planner/result.png)

## Acknowledgments
Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
