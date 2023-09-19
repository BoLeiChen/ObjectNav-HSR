import numpy as np
import matplotlib.pyplot as plt
from map_utils.constants import d3_40_colors_rgb
from PIL import Image
from einops import asnumpy

def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))

def segment_collision_check(wall_layer, point1, point2):
    length = np.linalg.norm((point2[0] - point1[0], point2[1] - point1[1]))
    vector = (point2 - point1) / length
    for i in range(int(length + 0.5)):
        px = int(point1[0] + vector[0] * i + 0.5)
        py = int(point1[1] + vector[1] * i + 0.5)
        if wall_layer[py][px] == 1:
            return True
    return False

def visualize_map(semmap, bg=1.0):
    n_cat = semmap.shape[0] - 2 # Exclude floor and wall
    def compress_semmap(semmap):
        c_map = np.zeros((semmap.shape[1], semmap.shape[2]))
        for i in range(semmap.shape[0]):
            c_map[semmap[i] > 0.] = i+1
        return c_map

    palette = [
            int(bg * 255), int(bg * 255), int(bg * 255), # Out of bounds
            230, 230, 230, # Free space
            77, 77, 77, # Obstacles
        ]

    palette += [c for color in d3_40_colors_rgb[:n_cat] for c in color.tolist()]
    semmap = asnumpy(semmap)
    c_map = compress_semmap(semmap)
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)

    return semantic_img


def plot_points(points, map):
    plt.subplot(1, 1, 1)
    global_map_vis = visualize_map(map[0:23, ...])
    plt.imshow(global_map_vis)
    for p in points:
        plt.scatter(p[0], p[1], s=1)
    plt.show()

def plot_path(path, map):
    plt.subplot(1, 1, 1)
    global_map_vis = visualize_map(map[0:23, ...])
    plt.imshow(global_map_vis)
    for p in path:
        plt.scatter(p[1], p[0], color='g', s=6)