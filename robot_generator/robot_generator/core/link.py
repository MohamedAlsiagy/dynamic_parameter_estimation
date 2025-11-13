import cadquery as cq
from math import pi, sqrt
import random
from robot_generator.utils.geometry_utils import rotated_rectangle_corners, Axis
from robot_generator.core.joint import create_revolute_end
from robot_generator.core.gripper import generate_gripper_female, generate_gripper_male

def create_link(length, diameter, com_bias, start_joint_type, end_joint_type , name , match_end_radius=False, cylendrify_base=False, cylendrify_ends=None, support_height_percent=None, equalize_mass=True , offset = (0 , 0) , s_xy_angle = 0 , e_xy_angle = 0 , is_base_link = False , is_last_link = False , include_gripper_if_last_link = (False , "x")):
    cylendrify_ends = cylendrify_ends if cylendrify_ends is not None else cylendrify_base
    com_bias = max(com_bias, -4 * diameter / 5) if diameter - abs(com_bias) < 0 else com_bias

    m_radius, s_radius, e_radius = diameter / 2, (diameter - com_bias) / 2, (diameter + com_bias) / 2
    
    offset_x, offset_y = offset
    # Create the main body of the link with optional cylindrical shaping
    if length != 0:
        if cylendrify_base:
            base = (cq.Workplane("XY").circle(s_radius).workplane(offset=length).center(offset_x,offset_y).circle(e_radius).loft(combine=True))
        else:
            # Convert angle to radians for calculation
            s_corners = rotated_rectangle_corners(2 * s_radius, 2 * s_radius , s_xy_angle)
            e_corners = rotated_rectangle_corners(2 * e_radius, 2 * e_radius , e_xy_angle)
            base = (
                cq.Workplane("XY")
                .polyline(s_corners).close()
                .workplane(offset=length)
                .center(offset_x, offset_y)  # Apply center here, before second rectangle
                .polyline(e_corners).close()
                .loft(combine=True)
            )
        base = base.translate((0, 0, -length / 2))
        link = base
    
    links = []
    axes = []
    s_subscript = "start_joint_axis"
    if is_base_link:
        s_hole_axis = Axis(name , s_subscript , (0,0,0) , (0,0,1)).translate((0, 0, -(length / 2)))
    else:
        s_end, s_height , s_hole_axis = create_revolute_end(start_joint_type, s_radius if match_end_radius else m_radius , name , s_subscript , flip=True, cylendrify=cylendrify_ends, support_height_percent=support_height_percent, equalize_mass=equalize_mass , xy_angle = s_xy_angle)
        s_end = s_end.translate((0, 0, -(length / 2 + s_height)))
        s_hole_axis = s_hole_axis.translate((0, 0, -(length / 2 + s_height)))
        link = s_end.union(link) if length != 0 else s_end
        
    e_subscript = "end_joint_axis" 
    if is_last_link:
        if include_gripper_if_last_link[0]:
            gripper, gripper_axis , finger_width , finger_base_height , travel = generate_gripper_female(e_radius if match_end_radius else m_radius, name , e_subscript , cylindrify=cylendrify_ends , axis = include_gripper_if_last_link[1])
            gripper = gripper.translate((offset_x, offset_y, length / 2))
            gripper_axis = gripper_axis.translate((offset_x, offset_y, length / 2))
            e_axis = gripper_axis
            link = gripper.union(link) if length != 0 or not is_base_link else gripper
            
            finger_1 , finger_1_axis = generate_gripper_male(finger_width, 4 * finger_base_height, "gripper_fingers", "finger_1", 
                                        cylindrify=cylendrify_ends , axis = include_gripper_if_last_link[1])
        
            finger_2 , finger_2_axis = generate_gripper_male(finger_width, 4 * finger_base_height, "gripper_fingers", "finger_2", 
                                        cylindrify=cylendrify_ends , axis = include_gripper_if_last_link[1] , mirror = True , shift_grips = True)
            
            finger_1_axis.info["travel"] = str(travel/2 - finger_width)
            finger_2_axis.info["travel"] = str(travel/2 - finger_width)
        else:
            e_axis = Axis(name , e_subscript , (0,0,0) , (0,0,1)).translate((offset_x, offset_y, length / 2))
    else:
        e_end, e_height , e_axis = create_revolute_end(end_joint_type, e_radius if match_end_radius else m_radius, name , e_subscript , female=False, cylendrify=cylendrify_ends, support_height_percent=support_height_percent, equalize_mass=equalize_mass , xy_angle = e_xy_angle)
        e_end = e_end.translate((offset_x, offset_y, length / 2))
        e_axis = e_axis.translate((offset_x, offset_y, length / 2))
        link = e_end.union(link) if length != 0 or not is_base_link else e_end

    link = link.translate(tuple(-s_hole_axis.origin))
    
    links.append(link)
    axes.append(s_hole_axis)
    axes.append(e_axis)

    if is_last_link and include_gripper_if_last_link[0]:
        links.append(finger_1)
        links.append(finger_2)
        axes.append(finger_1_axis)
        axes.append(finger_2_axis)
    
    return links , axes


def genertate_random(grid_size):
    # Create a new workplane for all links
    all_links = cq.Workplane("XY")
    # Generate random links and arrange them in a grid
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # Randomize parameters for each link
            length = random.uniform(3, 7)  # Random length between 1 and 3
            diameter = random.uniform(0.5, 2)  # Random diameter between 0.5 and 1.5
            com_bias = random.uniform(-0.25, 0.25)  # Random COM bias between -0.5 and 0.5
            start_joint_type = random.choice(["x", "y", "z"])  # Random start joint type
            end_joint_type = random.choice(["x", "y", "z"])  # Random end joint type
            match_end_radius = random.choice([True, False])
            cylendrify_base = random.choice([True, False])
            support_height_percent = True if not match_end_radius else random.choice([True, False])  # Random support height percentage
            equalize_mass = True
            offset = (random.uniform(-2, 2) , random.uniform(-2, 2))
            s_xy_angle = random.uniform(0 , 90)
            e_xy_angle = random.uniform(0 , 90)
            is_base_link = random.uniform(0 , 10)//9
            is_last_link = random.uniform(0 , 10)//9
            
            # Create a new random link with the generated parameters
            links , axes = create_link(
                name = "sample",
                length=length,
                diameter=diameter,
                com_bias=com_bias,
                start_joint_type=start_joint_type,
                end_joint_type=end_joint_type,
                match_end_radius=match_end_radius,
                cylendrify_base=cylendrify_base,
                support_height_percent=support_height_percent,
                equalize_mass=equalize_mass,
                offset = offset,
                s_xy_angle = s_xy_angle,
                e_xy_angle = e_xy_angle,
                is_base_link = is_base_link,
                is_last_link = is_last_link,
            )
            
            n_link = links[0]
            
            # Position the link on the grid
            x_pos = col * 6  # Adjust spacing between columns
            y_pos = row * 6  # Adjust spacing between rows
            
            for axis in axes:
                axis.translate((x_pos, y_pos, 0))
                axis.show()
            
            n_link = n_link.translate((x_pos, y_pos, 0))         
            # Add the link to the main object
            all_links = all_links.union(n_link)

    return all_links

#all_links = genertate_random(grid_size)
#show_object(all_links)
