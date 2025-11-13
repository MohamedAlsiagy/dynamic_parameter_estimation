import os
import subprocess
from tqdm import tqdm

def convert_stl_to_dae(stl_file, dae_file):
    # Step 1: Use Assimp command-line tool to convert STL to DAE
    try:
        subprocess.run(['assimp', 'export', stl_file, dae_file], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        # You can also silence the exception if you prefer not to print errors:
        pass
        # If you want to log errors instead of printing them:
        # logging.error(f"Error: Unable to convert STL file {stl_file} to DAE. {e}")

def open_and_edit_dae(dae_file , color):
    # Open the DAE file and read its contents
    with open(dae_file, 'r') as file:
        content = file.read()

    # Allow the user to manually edit the file (press Enter after editing)
    content = content.replace("ns0:" , "").replace(":ns0" , "")
    old_color = '<color sid="diffuse">1   1   1   1</color>'
    color = [round(c,3) for c in color]
    new_color = f'<color sid="diffuse">{color[0]}   {color[1]}   {color[2]}   1</color>'
    content = content.replace(old_color, new_color)

    # Save the content back to the file after the user is done editing
    with open(dae_file, 'w') as file:
        file.write(content)
    

def generate_dae_dir(stl_dir, dae_dir , colors):
    # Step 1: Ensure the destination directory exists
    if not os.path.exists(dae_dir):
        os.makedirs(dae_dir)

    files = os.listdir(stl_dir)
    ordered_files = sorted([files for files in files if files.endswith(".stl")])
    # Step 2: Loop through all STL files in the source directory
    for i, filename in tqdm(enumerate(ordered_files), desc="Converting STL to DAE"):
        color = colors[i]
        stl_file = os.path.join(stl_dir, filename)
        dae_file = os.path.join(dae_dir, filename.replace(".stl", ".dae"))
        convert_stl_to_dae(stl_file, dae_file)
        # After converting, open the DAE file for editing
        open_and_edit_dae(dae_file , color)