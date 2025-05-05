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

# Restliche Parameter übernehmbar?

FEET_SITES = [ # TODO: now this and the geoms are the same... idk if thats okay
    "l_foot",
    "r_foot",
]

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
