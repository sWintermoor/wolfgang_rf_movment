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
    "l_foot_site",
    "r_foot_site",
]

# TODO: Add the specific geoms (see above)
LEFT_FEET_GEOMS = [
    "l_foot_collision_1",
    "l_foot_collision_2",
    "l_foot_collision_3",
    "l_foot_collision_4",
    "l_foot_collision_5",
    "l_foot_collision_6",
    "l_foot_collision_7",
    "l_foot_collision_8",
    "l_foot_collision_9",
    "l_foot_collision_10",
    "l_foot_collision_11",
    "l_foot_collision_12",
    "l_foot_collision_13",
    "l_foot_collision_14",
    "l_foot_collision_15",
    "l_foot_collision_16",
    "l_foot_collision_17",
    "l_foot_collision_18",
    "l_foot_collision_19",
    "l_foot_collision_20",
    "l_foot_collision_21",
    "l_foot_collision_22",
    "l_foot_collision_23",
    "l_foot_collision_24",
    "l_foot_collision_25",
    "l_foot_collision_26",
    "l_foot_collision_27",
    "l_foot_collision_28",
    "l_foot_collision_29",
    "l_foot_collision_30",
    "l_foot_collision_31",
    "l_foot_collision_32",
    "l_foot_collision_33",
    "l_foot_collision_34",
    "l_foot_collision_35",
    "l_foot_collision_36",
    "l_foot_collision_37",
    "l_foot_collision_38",
    "l_foot_collision_39",
    "l_foot_collision_40",
    "l_foot_collision_41",
]

RIGHT_FEET_GEOMS = [
    "r_foot_collision_1",
    "r_foot_collision_2",
    "r_foot_collision_3",
    "r_foot_collision_4",
    "r_foot_collision_5",
    "r_foot_collision_6",
    "r_foot_collision_7",
    "r_foot_collision_8",
    "r_foot_collision_9",
    "r_foot_collision_10",
    "r_foot_collision_11",
    "r_foot_collision_12",
    "r_foot_collision_13",
    "r_foot_collision_14",
    "r_foot_collision_15",
    "r_foot_collision_16",
    "r_foot_collision_17",
    "r_foot_collision_18",
    "r_foot_collision_19",
    "r_foot_collision_20",
    "r_foot_collision_21",
    "r_foot_collision_22",
    "r_foot_collision_23",
    "r_foot_collision_24",
    "r_foot_collision_25",
    "r_foot_collision_26",
    "r_foot_collision_27",
    "r_foot_collision_28",
    "r_foot_collision_29",
    "r_foot_collision_30",
    "r_foot_collision_31",
    "r_foot_collision_32",
    "r_foot_collision_33",
    "r_foot_collision_34",
    "r_foot_collision_35",
    "r_foot_collision_36",
    "r_foot_collision_37",
    "r_foot_collision_38",
    "r_foot_collision_39",
    "r_foot_collision_40",
    "r_foot_collision_41",
]

def task_to_xml(task_name: str) -> epath.Path: 
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_FLAT_TERRAIN_XML, # TODO: change later
  }[task_name]

FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "torso"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
