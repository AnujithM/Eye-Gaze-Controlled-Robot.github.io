import cv2
import numpy as np
import threading
import requests
import math
import time
import supervision as sv
from inference.models.utils import get_model

model = get_model(model_id="hand-segmentation-qqx45/7", api_key="K8vrKwiO7UYL6KrQQFKc")

# Create supervision annotators
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

# Global variables
previous_red_centroids_count = 0

# Function to fetch MJPEG frames over HTTP and decode them using OpenCV
def fetch_and_decode_frames(url):
    global centroid_coordinates, goal_location
    while True:
        try:
            response = requests.get(url)
            if response is not None:
                img_array = bytearray(response.content)
                frame = cv2.imdecode(np.array(img_array), -1)
                if not process_frame(frame):
                    break
            else:
                print("Failed to fetch frames. Response is None.")
        except Exception as e:
            print(f"Error fetching MJPEG frames: {e}")

def heuristic(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def find_path_dynamic_a_star(source, goal, adjacency_list):
    visited = set()
    queue = [(source, [source], 0)]
    heuristic_cost = {source: heuristic(source, goal)}
    
    while queue:
        current_node, path, cost = queue.pop(0)
        if current_node == goal:
            return path
        if current_node not in visited:
            visited.add(current_node)
            for neighbor in adjacency_list[current_node]:
                new_cost = cost + 1
                new_heuristic_cost = new_cost + heuristic(neighbor, goal)
                if neighbor not in heuristic_cost or new_heuristic_cost < heuristic_cost[neighbor]:
                    heuristic_cost[neighbor] = new_heuristic_cost
                    queue.append((neighbor, path + [neighbor], new_cost))
            queue.sort(key=lambda x: heuristic_cost[x[0]])
    return None

def display_frame_rate(frame, start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    frame_rate = 1.0 / elapsed_time
    cv2.putText(frame, f"FPS: {int(frame_rate)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def process_frame(frame):
    global centroid_coordinates, goal_location, source_index, previous_red_centroids_count

    start_time = time.time()
    results = model.infer(frame)
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))
    annotated_frame = mask_annotator.annotate(scene=frame, detections=detections)
    adjacency_list = {(i, j): [] for i in range(20) for j in range(20)}
    red_centroids = set()
    rows, cols, _ = annotated_frame.shape
    row_step = rows // 20
    col_step = cols // 20
    for i in range(20):
        for j in range(20):
            x = int((j + 0.5) * col_step)
            y = int((i + 0.5) * row_step)
            centroid_coordinates[f"({i},{j})"] = (x, y)
            if x >= 0 and x < cols and y >= 0 and y < rows:
                overlap = False
                for detection in detections.xyxy:
                    if detection[0] <= x <= detection[2] and detection[1] <= y <= detection[3]:
                        overlap = True
                        break
                if not overlap:
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            new_i, new_j = i + dx, j + dy
                            if 0 <= new_i < 20 and 0 <= new_j < 20:
                                adjacency_list[(i, j)].append((new_i, new_j))
                else:
                    red_centroids.add((i, j))
            if x >= 0 and x < cols and y >= 0 and y < rows:
                if (i, j) in red_centroids:
                    cv2.circle(annotated_frame, (x, y), 2, (0, 0, 255), -1)
                else:
                    cv2.circle(annotated_frame, (x, y), 2, (255, 0, 0), -1)
            else:
                cv2.circle(annotated_frame, (x, y), 2, (255, 0, 0), -1)
            if i > 0:
                cv2.line(annotated_frame, (j * col_step, i * row_step), ((j + 1) * col_step, i * row_step), (255, 255, 255), 1)
            if j > 0:
                cv2.line(annotated_frame, (j * col_step, i * row_step), (j * col_step, (i + 1) * row_step), (255, 255, 255), 1)
    
    goal_location = centroid_coordinates["(0,14)"]
    if goal_location is not None:
        cv2.circle(annotated_frame, goal_location, 5, (0, 255, 0), -1)
        if source_index is not None:
            cv2.circle(annotated_frame, centroid_coordinates[f"({source_index[0]},{source_index[1]})"], 5, (0, 255, 255), -1)
            goal_index = (goal_location[1] // row_step, goal_location[0] // col_step)
            path = find_path_dynamic_a_star(source_index, goal_index, adjacency_list)
            if path:
                for index in range(len(path) - 1):
                    start_x, start_y = centroid_coordinates[f"({path[index][0]},{path[index][1]})"]
                    end_x, end_y = centroid_coordinates[f"({path[index + 1][0]},{path[index + 1][1]})"]
                    cv2.line(annotated_frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 1)
                time.sleep(1)
                if len(path) > 1:
                    source_index = path[1]
                else:
                    source_index = (12, 0)

    # Check for hand position change
    current_red_centroids_count = len(red_centroids)
    if current_red_centroids_count != previous_red_centroids_count:
        print("Hand position changed.")
    previous_red_centroids_count = current_red_centroids_count

    display_frame_rate(annotated_frame, start_time)
    cv2.imshow('Segmented Frame', annotated_frame)
    return cv2.waitKey(1) & 0xFF != ord('q')

centroid_coordinates = {}
source_index = (12, 0)
url = "http://192.168.227.165:8080/shot.jpg"
fetch_thread = threading.Thread(target=fetch_and_decode_frames, args=(url,))
fetch_thread.start()
fetch_thread.join()
cv2.destroyAllWindows()
