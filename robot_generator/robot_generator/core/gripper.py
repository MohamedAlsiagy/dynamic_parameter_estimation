import cadquery as cq
from math import pi, sqrt
from robot_generator.utils.geometry_utils import Axis

def generate_gripper_female(radius, name, subscript, support_height_percent=None, 
                            slot_thickness=None, slot_ratio=3, slot_width_ratio=1, 
                            wall_thickness_percentage=0.20, axis="x", cylindrify=False):
    
    # Calculate derived dimensions
    square_size = radius * 2
    if slot_thickness is None:
        slot_thickness = radius
    if support_height_percent is None:
        support_height_percent = 0.5
    support_height = radius * support_height_percent
    diameter = radius * 2
    slot_width = diameter * slot_width_ratio
    wall_thickness = wall_thickness_percentage * slot_width
    
    # Define base sketch (circle for cylindrify or square for non-cylindrical)
    if cylindrify:
        base_sketch = cq.Workplane("XY").circle(radius)
    else:
        base_sketch = cq.Workplane("XY").rect(diameter, diameter)
    
    # Function to create a slot (either circular or rectangular)
    def create_slot(wp, width, height):
        if cylindrify:
            return wp.slot2D(width, height)
        return wp.rect(width, height)
    
    # Create the gripper base if support height is non-zero
    if support_height != 0:
        base_shape = (
            create_slot(base_sketch.workplane(offset=support_height), slot_width * slot_ratio, slot_width)
            .loft(combine=True)
        )
    
    # Create a primary slot and extrude it
    primary_slot = (
        create_slot(cq.Workplane("XY"), slot_width * slot_ratio, slot_width)
        .extrude(slot_thickness)
    ).translate((0, 0, support_height))
    
    # Combine primary slot with the base if support height is non-zero
    primary_slot = primary_slot.union(base_shape) if support_height != 0 else primary_slot

    # Create a negative slot with wall thickness applied and extrude
    finger_width = slot_width - 2 * wall_thickness
    finger_base_height = slot_thickness - wall_thickness
    travel = slot_width * slot_ratio - 2 * wall_thickness
    
    negative_slot = (
        create_slot(cq.Workplane("XY"), travel , finger_width)
        .extrude(finger_base_height)
    ).translate((0, 0, support_height + wall_thickness))
    
    # Cut the negative slot from the primary slot
    gripper = primary_slot.cut(negative_slot)
    
    # Create the gripper axis
    gripper_axis = (
        Axis(name, subscript, (0, 0, 0), (1, 0, 0))
        .translate((0, 0, support_height + wall_thickness))
        .rotate((0, 0, 1), (0, 0, 0), (axis == "x") * 90)
    )
    
    # Show the axis and rotate the gripper if necessary
    gripper_axis.show()
    gripper = gripper.rotate((0, 0, 1), (0, 0, 0), (axis == "x") * 90)

    return gripper, gripper_axis , finger_width , finger_base_height , travel

def generate_gripper_male(finger_width, finger_height, name, subscript, 
                           axis="x", 
                           cylindrify=False, mirror=False, grip_count=24, 
                           shift_grips=False, include_grips=False):
    
    # Base shape generation
    if cylindrify:
        base_shape = cq.Workplane("XY").circle(finger_width / 2).extrude(finger_height - finger_width / 2)
        dome = cq.Workplane("XY").sphere(finger_width / 2).translate((0, 0, finger_height - finger_width / 2))
        dome_half = dome.cut(cq.Workplane("XY").circle(finger_width / 2).extrude(finger_width / 2).translate((0, 0, finger_height - finger_width)))
        base_shape = base_shape.union(dome_half)

        box = cq.Workplane("XY").rect(finger_width, finger_width / 2).extrude(finger_height).translate((0, finger_width / 4, 0))
        base_shape = base_shape.cut(box).union(box)
    else:
        base_shape = cq.Workplane("XY").rect(finger_width, finger_width).extrude(finger_height)
    
    # Adding grips
    if include_grips:
        side_length = finger_height / grip_count
        height = (sqrt(3) / 2) * side_length
        grips_space = cq.Workplane("XY").rect(finger_width, height).extrude(finger_height).translate((0 , finger_width/2-height/2 , 0))
        base_shape = base_shape.cut(grips_space)
        
        for i in range(grip_count + shift_grips):
            points = [
                (0, 0),                            # Bottom-left corner
                (side_length, 0),                  # Bottom-right corner
                (side_length / 2, height)          # Top corner
            ]
            grip = (cq.Workplane("ZY")
                    .transformed(offset=(0, finger_width / 2 - height, -finger_width / 2))
                    .polyline(points)
                    .close()
                    .extrude(finger_width)
                   ).translate((0, 0, (i - shift_grips / 2) * side_length))
            base_shape = base_shape.union(grip)
            
    # Complementary shape
    if cylindrify:
        complementary_shape = cq.Workplane("XY").rect(finger_width, 2 * finger_width).extrude(finger_height - finger_width / 2)
        half_cylinder = (
            cq.Workplane("XY")
            .circle(finger_width / 2)
            .extrude(2 * finger_width)
            .translate((0, 0, -finger_width))
            .cut(cq.Workplane("ZY").rect(2 * finger_width, finger_width).extrude(finger_width))
            .rotate((0, 1, 0), (0, 0, 0), 90)
            .rotate((0, 0, 1), (0, 0, 0), (axis == "y") * 90)
            .translate((0, 0, finger_height - finger_width / 2))
        )
        complementary_shape = complementary_shape.union(half_cylinder)
    else:
        complementary_shape = cq.Workplane("XY").rect(finger_width, 2 * finger_width).extrude(finger_height)

    # Negative shape to cut out the complementary shape
    negative_shape = cq.Workplane("XY").rect(2 * finger_width, 2 * finger_width).extrude(2 * finger_height).translate((0, 0, -finger_height / 2))
    negative_shape = negative_shape.cut(complementary_shape)
    
    base_shape = base_shape.cut(negative_shape)
    base_shape = base_shape.translate((0 , -finger_width/2 , 0))

    if include_grips:
        base_shape = base_shape.translate((0 , height/2 , 0))
    # Axis for finger (optional)
    finger_axis = (Axis(name, subscript, (0, 0, 0), (1, 0, 0))
                   .rotate((0, 0, 1), (0, 0, 0), (axis == "x") * -90)
    )
    
    # Rotate the base shape
    base_shape = base_shape.rotate((0, 0, 1), (0, 0, 0), (axis == "y") * 90)

    # Apply mirror if requested
    if mirror:
        base_shape = base_shape.rotate((0, 0, 1), (0, 0, 0), 180)
    else:
        finger_axis = finger_axis.rotate((0, 0, 1), (0, 0, 0), 180)

    # Return the final finger shape and its axis
    finger = base_shape
    finger_axis.show()

    return finger, finger_axis