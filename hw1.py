"""
Note: This code was written with the assistance of chatGPT
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the cross product of two vectors
def cross_product(v1, v2):
    return np.cross(v1, v2)

# Function to check if a point lies on a line segment
def is_on_segment(p, v1, v2):
    # Check if the point is within the bounding box of the segment
    return (min(v1[0], v2[0]) <= p[0] <= max(v1[0], v2[0]) and
            min(v1[1], v2[1]) <= p[1] <= max(v1[1], v2[1]))

# Function to convert Cartesian coordinates to homogeneous coordinates
def to_homogeneous(v):
    return np.array([v[0], v[1], 1])

# Function to compute if the aim is correct
def check_aim(aim_angle, v1, v2, v3):
    # Convert aiming line to homogeneous coordinates
    l = np.array([np.tan(np.radians(aim_angle)), -1, 0])
    
    # Convert triangle vertices to homogeneous coordinates
    v1_h = to_homogeneous(v1)
    v2_h = to_homogeneous(v2)
    v3_h = to_homogeneous(v3)
    
    # Compute lines corresponding to triangle edges
    l12 = cross_product(v1_h, v2_h)
    l23 = cross_product(v2_h, v3_h)
    l31 = cross_product(v3_h, v1_h)
    
    # Find intersection points
    p12 = cross_product(l, l12)
    p23 = cross_product(l, l23)
    p31 = cross_product(l, l31)
    
    # Normalize homogeneous coordinates (convert to Cartesian)
    if p12[2] != 0:
        p12 /= p12[2]
    if p23[2] != 0:
        p23 /= p23[2]
    if p31[2] != 0:
        p31 /= p31[2]
    
    # Check if any intersection point lies on the corresponding segment
    hit = (is_on_segment(p12[:2], v1, v2) or 
           is_on_segment(p23[:2], v2, v3) or 
           is_on_segment(p31[:2], v3, v1))
    
    return hit, p12, p23, p31

# Function to plot the scenario
def plot_scenario(aim_angle, v1, v2, v3, hit):
    plt.figure(figsize=(6, 6))
    plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'ro-')
    plt.plot([v2[0], v3[0]], [v2[1], v3[1]], 'ro-')
    plt.plot([v3[0], v1[0]], [v3[1], v1[1]], 'ro-')
    
    x_vals = np.linspace(-10, 10, 400)
    y_vals = np.tan(np.radians(aim_angle)) * x_vals
    plt.plot(x_vals, y_vals, 'b--', label=f'Aim Line (α = {aim_angle}°)')
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    if hit:
        plt.title("Hit! The aim intersects the triangle.")
    else:
        plt.title("Miss! The aim does not intersect the triangle.")
    
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate random triangles and aim angles
np.random.seed(0)
for _ in range(5):
    # Random triangle vertices with the triangle pointing downwards
    v1 = np.array([np.random.uniform(-5, 5), np.random.uniform(5, 10)])
    v2 = np.array([v1[0] + np.random.uniform(2, 4), v1[1]])
    v3 = np.array([(v1[0] + v2[0]) / 2, v1[1] - np.random.uniform(2, 4)])   
    # Random aim angle
    aim_angle = np.random.uniform(-60, 60) 
    # Check if the aim is correct
    hit, _, _, _ = check_aim(aim_angle, v1, v2, v3)
    # Plot the scenario
    plot_scenario(aim_angle, v1, v2, v3, hit)
