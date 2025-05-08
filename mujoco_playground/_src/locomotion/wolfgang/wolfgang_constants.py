# Work in progress# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for Wolfgang."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "wolfgang"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "wolfgang_scene.xml"
)

# Currently only body elements for l_foot/r_foot Wolfgang -> Reasonable to create site and geoms for
# these body elements, because in joystick.py we use methods that are specifically designed for working 
# with sites and geoms  
# -> site: Usage in joystick: Using for food position; 
# TODO: Create a site in l_foot/r_foot with pos arguments, size(?) e.g. 'l_foot_size' -> change objtype to 'size'
# -> geom: Usage in joystick: Collision detection (contype and conaffinity are important for collision detection); 
# TODO: Add names for every geom in l_foot/r_foot with class='collision' e.g.: l_foot1, l_foot2, l_foot3... 

# TODO: Create new site called l_foot_size and r_foot_size (see above)
FEET_SITES = [ # TODO: now this and the geoms are the same... idk if thats okay
    "l_foot",
    "r_foot",
]

# TODO: Add the specific geoms (see above)
LEFT_FEET_GEOMS = [
    "l_foot",
]

RIGHT_FEET_GEOMS = [
    "r_foot",
]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "torso"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
