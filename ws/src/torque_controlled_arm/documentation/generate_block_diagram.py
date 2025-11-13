import graphviz

def generate_block_diagram():
    dot = graphviz.Digraph(comment='Dataset Generation Process', graph_attr={'rankdir': 'LR', 'splines': 'ortho', 'nodesep': '0.8', 'ranksep': '1.0'}, node_attr={'shape': 'box', 'style': 'filled', 'fillcolor': '#E0F2F7', 'fontname': 'Helvetica'}, edge_attr={'fontname': 'Helvetica', 'fontsize': '10'}) # Light blue background

    # Nodes
    dot.node('A', 'Settings (settings.yaml)', fillcolor='#FFF9C4') # Light yellow
    dot.node('B', 'Original URDFs', fillcolor='#CFD8DC') # Blue Grey
    dot.node('C', 'gazebo_adapter.py', fillcolor='#C8E6C9') # Light Green
    dot.node('D', 'Modified URDFs (robotGA.urdf)', fillcolor='#B2EBF2') # Cyan
    dot.node('E', 'data_generator_node.py', fillcolor='#C8E6C9') # Light Green
    dot.node('F', 'Dynamic Parameters (YAML)', fillcolor='#FFF9C4') # Light yellow
    dot.node('G', 'Gazebo Simulation', fillcolor='#BBDEFB') # Light Blue
    dot.node('H', 'combined_controller_node.py', fillcolor='#C8E6C9') # Light Green
    dot.node('I', 'Trajectory Data (CSV per robot)', fillcolor='#F8BBD0') # Pink
    dot.node('J', 'aggregate_csv.py', fillcolor='#C8E6C9') # Light Green
    dot.node('K', 'Aggregated Trajectory Data (CSV)', fillcolor='#F8BBD0') # Pink

    # Edges
    dot.edge('A', 'C', label='Reads config')
    dot.edge('B', 'C', label='Input URDFs')
    dot.edge('C', 'D', label='Generates')
    dot.edge('D', 'E', label='Reads modified URDFs')
    dot.edge('E', 'F', label='Extracts & Saves')
    dot.edge('D', 'G', label='Spawns robot')
    dot.edge('G', 'H', label='Joint States, Link States')
    dot.edge('H', 'G', label='Torque Commands, Reset Signal')
    dot.edge('H', 'I', label='Records & Saves')
    dot.edge('I', 'J', label='Inputs individual CSVs')
    dot.edge('J', 'K', label='Outputs single CSV')

    dot.render('dataset_generation_process', view=False, format='png')
    print('Block diagram generated as dataset_generation_process.png')

if __name__ == '__main__':
    generate_block_diagram()


