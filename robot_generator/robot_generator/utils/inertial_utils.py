from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

# Load the STEP file and extract mass properties
def get_mass_inertia_from_step(step_file_path):
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file_path)
    if status == 1:
        step_reader.TransferRoots()
        shape = step_reader.OneShape()
        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)

        density = 600
        mass = props.Mass()* density
        center_of_mass = props.CentreOfMass() 
        inertia_tensor = props.MatrixOfInertia() * density
        
        inertia = {
            "mass": mass,
            "center_of_mass": (center_of_mass.X(), center_of_mass.Y(), center_of_mass.Z()),
            "ixx": inertia_tensor.Value(1, 1),
            "ixy": inertia_tensor.Value(1, 2),
            "ixz": inertia_tensor.Value(1, 3),
            "iyy": inertia_tensor.Value(2, 2),
            "iyz": inertia_tensor.Value(2, 3),
            "izz": inertia_tensor.Value(3, 3)
        }
        return inertia
    else:
        print(f"Failed to read STEP file {step_file_path}")
        return None
