import cadquery as cq
from math import pi, sqrt
from robot_generator.utils.geometry_utils import Axis

def create_XY_revolute_end(radius, name , subscript , support_height_percent=None, segment_width=None, hole_radius=None, female=True, axis="x", flip=False, cylendrify=False, equalize_mass=False , xy_angle = 0):
    # Set default parameters if not provided
    segment_width = segment_width or radius * 2 / 3
    hole_radius = hole_radius or radius / 3
    if support_height_percent == None:
        support_height_percent = 1 / 2
        
    support_height = radius * support_height_percent
    square_size = radius * 2
    square_height = radius + support_height
    half_cylinder_height = segment_width * 3
    
    if not cylendrify:
        # Create the extruded square (a cuboid)
        square = cq.Workplane("XY").rect(square_size, half_cylinder_height).extrude(square_height)
    
        # Create the half-cylinder and place it on top of the extruded square
        half_cylinder = (
            cq.Workplane("XY")
            .circle(radius)
            .extrude(half_cylinder_height)
            .translate((0, 0, -half_cylinder_height / 2))
            .cut(
                cq.Workplane("ZY")
                .rect(2 * radius, half_cylinder_height)
                .extrude(2 * radius)
            )
            .rotate((0, 1, 0), (0, 0, 0), 90)
            .rotate((0, 0, 1), (0, 0, 0), (axis == "x") * 90)
            .translate((0, 0, square_height))
        )
        result = square.union(half_cylinder)
    else:
        # Create a cylinder
        cylinder = cq.Workplane("XY").circle(radius).extrude(square_height)
    
        # Create a dome by cutting a hemisphere from a sphere
        dome = cq.Workplane("XY").sphere(radius).translate((0, 0, square_height))
        
        # Cut half of the sphere (to create a dome)
        dome_half = dome.cut(cq.Workplane("XY").circle(radius).extrude(radius).translate((0, 0, square_height- radius)))

        result = cylinder.union(dome_half)
    
    # Create the cutout shape in the square
    if female:
        if equalize_mass:
            if cylendrify:
                segment_width = 2 * (1 - 0.65270) * radius
            else:
                segment_width = radius
        cutout = (
            cq.Workplane("ZX")
            .box(square_size, segment_width, square_size)
            .translate((0, 0, square_height))
        )
    else:
        if equalize_mass:
            if cylendrify:
                segment_width =  0.65270 * radius
            else:
                segment_width = radius/2
        translate_value = radius - segment_width / 2
        cutout_R = (
            cq.Workplane("ZX")
            .box(square_size, segment_width, square_size)
            .translate((-translate_value, 0, square_height))
        )
        cutout_L = (
            cq.Workplane("ZX")
            .box(square_size, segment_width, square_size)
            .translate((translate_value, 0, square_height))
        )
        cutout = cutout_R.union(cutout_L)

    cutout = cutout.rotate((0, 0, 1), (0, 0, 0), (axis == "x") * 90)
    result = result.cut(cutout)
    
    hole_axis  = (
                Axis(name , subscript , (0,0,0) , (1,0,0))
                .translate((0, 0, square_height))
                .rotate((0, 0, 1), (0, 0, 0), (axis == "x") * 90)
            )
    
    # Create and subtract the hole
    cutout_hole = (
        cq.Workplane("ZY")
        .circle(hole_radius)
        .extrude(square_size)
        .translate((square_size / 2, 0, square_height))
        .rotate((0, 0, 1), (0, 0, 0), (axis == "x") * 90)
    )
    xy_revolute_end = result.cut(cutout_hole)

    total_height = square_height + radius

    # Optional flipping
    if flip:
        mid = total_height / 2
        xy_revolute_end = xy_revolute_end.rotate((1, 0, mid), (0, 0, mid), 180)
        hole_axis = hole_axis.rotate((1, 0, mid), (0, 0, mid), 180)
    
    xy_revolute_end = xy_revolute_end.rotate((0 , 0 , 0) , (0 , 0 , 1) , xy_angle)
    hole_axis = hole_axis.rotate((0 , 0 , 0) , (0 , 0 , 1) , xy_angle)

    # hole axis correction always keep its angle from 0 to pi
    direction = hole_axis.direction
    if direction[1] < 0 or (direction[1] == 0 and direction[0] < 0):
        hole_axis = hole_axis.rotate((0 , 0 , 0) , (0 , 0 , 1) , 180)

    return xy_revolute_end, total_height , hole_axis


def create_Z_revolute_end(radius, name , subscript ,  support_height_percent=None, segment_width=None, hole_radius=None, female=True, axis="x", flip=False, cylendrify=False, equalize_mass=False, xy_angle = 0):
    segment_width = segment_width or radius * 2 / 3
    hole_radius = hole_radius or radius / 3
    if support_height_percent == None:
        support_height_percent = 1 / 2

    support_height = radius * support_height_percent
    square_size = radius * 2
    half_cylinder_height = segment_width * 3

    # Create the extruded square if support height is specified
    if support_height != 0:
        square = cq.Workplane("XY").rect(square_size, half_cylinder_height).extrude(support_height)

    # Create the cylinder and optionally adjust radius to equalize mass
    if equalize_mass:
        hole_radius = radius / sqrt(2)
    
    hole_axis  = Axis(name , subscript , (0,0,0) , (0,0,1))
    cylinder = cq.Workplane("XY").circle(hole_radius).extrude(segment_width)

    if female:
        complementry_cylinder = cq.Workplane("XY").circle(radius).extrude(segment_width)
        complementry_cylinder = complementry_cylinder.cut(cylinder).translate((0, 0, support_height))
        z_revolute_end = square.union(complementry_cylinder) if support_height != 0 else complementry_cylinder
    else:
        hole_axis = hole_axis.translate((0, 0, segment_width))
        cylinder = cylinder.translate((0, 0, support_height))
        z_revolute_end = square.union(cylinder) if support_height != 0 else cylinder
    
    hole_axis = hole_axis.translate((0, 0, support_height))
    total_height = segment_width + support_height

    if cylendrify:
        complementarty_square = cq.Workplane("XY").rect(square_size, half_cylinder_height).extrude(total_height)
        c_radius = max(square_size, half_cylinder_height) / 2
        c_cylinder = cq.Workplane("XY").circle(c_radius).extrude(total_height)
        complementarty_square = complementarty_square.cut(c_cylinder)
        z_revolute_end = z_revolute_end.cut(complementarty_square)

    if flip:
        mid = total_height / 2
        z_revolute_end = z_revolute_end.rotate((1, 0, mid), (0, 0, mid), 180)
        hole_axis = hole_axis.rotate((1, 0, mid), (0, 0, mid), 180)
        
    z_revolute_end = z_revolute_end.rotate((0 , 0 , 0) , (0 , 0 , 1) , xy_angle)
    hole_axis = hole_axis.rotate((0 , 0 , 0) , (0 , 0 , 1) , xy_angle)
    
    return z_revolute_end, total_height , hole_axis

def create_revolute_end(axis, radius , name , subscript , support_height_percent=None, segment_width=None, hole_radius=None, female=True, flip=False, cylendrify=False, equalize_mass=False , xy_angle = 0):
    if axis == "z":
        return create_Z_revolute_end(radius, name , subscript , support_height_percent, segment_width, hole_radius, female, axis, flip, cylendrify, equalize_mass, xy_angle)
    return create_XY_revolute_end(radius, name , subscript, support_height_percent, segment_width, hole_radius, female, axis, flip, cylendrify, equalize_mass , xy_angle)
