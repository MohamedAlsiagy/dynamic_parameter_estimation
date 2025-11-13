import cadquery as cq
import numpy as np
import json
import os
from math import radians, cos, sin

class Axis:
    def __init__(self, name , subscript , origin=(0, 0, 0), direction=(1, 0, 0)):
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float) / np.linalg.norm(direction)
        self.name = name
        self.subscript = subscript
        self.info = dict()
        
    def translate(self, translation_vector):
        """Translate the axis by the given 3D vector."""
        self.origin += np.array(translation_vector)
        return self

    def rotate(self, axisStartPoint, axisEndPoint, angleDegrees):
        """Rotate the axis around the axis defined by two points and an angle."""
        # Convert degrees to radians
        angleRadians = np.radians(angleDegrees)
        
        # Calculate the axis of rotation
        rotation_axis = np.array(axisEndPoint) - np.array(axisStartPoint)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Rodrigues' rotation formula
        def rodrigues_rotation(v, axis, angle):
            """Rotate vector v around axis by angle using Rodrigues' formula."""
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            return (v * cos_angle + 
                    np.cross(axis, v) * sin_angle + 
                    axis * np.dot(axis, v) * (1 - cos_angle))

        # Rotate the origin and direction using Rodrigues' formula
        self.origin = rodrigues_rotation(self.origin - np.array(axisStartPoint), rotation_axis, angleRadians) + np.array(axisStartPoint)
        self.direction = rodrigues_rotation(self.direction, rotation_axis, angleRadians)
        
        return self
    
    def _get_cq_axis(self , end_point):
        axis_object = cq.Workplane("XY").moveTo(self.origin[0], self.origin[1]).lineTo(end_point[0], end_point[1]).translate((self.origin[0], self.origin[1], self.origin[2]))
        return axis_object
    
    def show(self, length=1 , show_negative = False):
        # Step 5: Create the axis line from the origin along the direction
        end_point = self.origin + self.direction * length
        axis_line = cq.Edge.makeSpline([cq.Vector(*pnt) for pnt in [self.origin , end_point]])
    
        # Step 6: Show the axis line on the rotated workplane
        try:
            show_object(axis_line)
        except:
            pass

        if show_negative:
            # For the opposite direction (negative axis direction), do the same
            n_end_point = self.origin - self.direction * length
            n_axis_line = cq.Edge.makeSpline([cq.Vector(*pnt) for pnt in [self.origin, n_end_point]])
        
            # Show the negative direction axis line
            try:
                show_object(n_axis_line)
            except:
                pass

    def export_json(self, parent_dir="json"):
        """Export the axis data to a JSON format, merging with existing data if present."""
        axis_data = {
            self.subscript: {
                "origin": self.origin.tolist(),
                "direction": self.direction.tolist(),
                **self.info
            },
            
        }
        print(axis_data)
        # Create the directory if it does not exist
        os.makedirs(parent_dir, exist_ok=True)
        
        # Construct the full file path
        filename = f"{self.name}.json"
        file_path = os.path.join(parent_dir, filename)
        
        # Load existing data if the file exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = {}
    
        # Merge new data with existing data
        existing_data.update(axis_data)
        
        # Save the updated data back to the file
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
        
        # print(f"Data saved to {file_path}")

    def __repr__(self):
        return f"Axis(origin={self.origin.tolist()}, direction={self.direction.tolist()})"


def rotated_rectangle_corners(width, length, angle_deg):
    # Convert angle to radians for calculation
    angle_rad = radians(angle_deg)
    
    # Half-width and half-length
    half_width = width / 2
    half_length = length / 2
    
    # Calculate the corner points of the rotated rectangle
    corners = [
        (half_width * cos(angle_rad) - half_length * sin(angle_rad),
         half_width * sin(angle_rad) + half_length * cos(angle_rad)),
        (-half_width * cos(angle_rad) - half_length * sin(angle_rad),
         -half_width * sin(angle_rad) + half_length * cos(angle_rad)),
        (-half_width * cos(angle_rad) + half_length * sin(angle_rad),
         -half_width * sin(angle_rad) - half_length * cos(angle_rad)),
        (half_width * cos(angle_rad) + half_length * sin(angle_rad),
         half_width * sin(angle_rad) - half_length * cos(angle_rad)),
    ]
    
    return corners