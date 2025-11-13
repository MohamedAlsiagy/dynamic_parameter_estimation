import os
from glob import glob
from setuptools import setup
from setuptools import find_packages

package_name = 'torque_controlled_arm'
robots_dir = 'robots'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files = [
    ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
    ('share/' + package_name, ['package.xml']),

    ('share/' + package_name + '/launch', glob('launch/*.py')),
    ('share/' + package_name + '/config', glob('config/*')),
    ('share/' + package_name + '/srv', glob('srv/*.srv')),
    
    ('share/' + package_name + '/world', glob('world/*')),
    ('share/' + package_name + '/robot_99', glob('robot_99/*.urdf')),
    ('share/' + package_name + '/robot_99/stl', glob('robot_99/stl/*.stl')),
    ('share/' + package_name + '/robot_99/dae', glob('robot_99/dae/*.dae')),
    
    ] + [
        (f'share/{package_name}/robots/{robot}/stl', glob(f'{robots_dir}/{robot}/stl/*'))
        for robot in os.listdir(robots_dir)
        if os.path.isdir(f'{robots_dir}/{robot}/stl')
    ] + [
        (f'share/{package_name}/robots/{robot}/dae', glob(f'{robots_dir}/{robot}/dae/*'))
        for robot in os.listdir(robots_dir)
        if os.path.isdir(f'{robots_dir}/{robot}/dae')
    ] + [
        (f'share/{package_name}/robots/{robot}', glob(f'{robots_dir}/{robot}/*.urdf'))
        for robot in os.listdir(robots_dir)
        if any(f.endswith('.urdf') for f in os.listdir(f'{robots_dir}/{robot}'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammed',
    maintainer_email='mohammed@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pid = torque_controlled_arm.pid:main',
            'target_gen = torque_controlled_arm.target_gen:main',

            'controller_node = torque_controlled_arm.controller_node:main',
        ],
    },
)
