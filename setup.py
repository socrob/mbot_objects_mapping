#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# for your packages to be recognized by python
d = generate_distutils_setup(
 packages=['mbot_objects_mapping_ros'],
 package_dir={'mbot_objects_mapping_ros': 'ros/src/mbot_objects_mapping_ros'}
)

setup(**d)
