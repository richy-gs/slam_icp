import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'slam_icp'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
    ],
    install_requires=['setuptools'],
    extras_require={'test': ['pytest']},
    zip_safe=True,
    maintainer='richy-gs',
    maintainer_email='richy-gs@users.noreply.github.com',
    description='SLAM 2D con ICP y MCL para ROS 2 Humble, sin paquetes de '
                'navegacion externos.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'slam_icp_node = slam_icp.slam_node:main',
            'wheel_odometry = slam_icp.robot.odometry:main',
            'scan_republisher = slam_icp.robot.scan_republisher:main',
        ],
    },
)
