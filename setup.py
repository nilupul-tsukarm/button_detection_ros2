from setuptools import setup
import os
from glob import glob

package_name = 'button_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='ROS 2 button detection node using RealSense',
    license='MIT',
    entry_points={
        'console_scripts': [
            'button_detection_node = button_detection.button_detection_node:main'
        ],
    },
)

