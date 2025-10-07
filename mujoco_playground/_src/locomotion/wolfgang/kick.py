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
"""Joystick task for Berkeley Humanoid."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.wolfgang import base as wolfgang_base
from mujoco_playground._src.locomotion.wolfgang import wolfgang_constants as consts

#TODO: self._init_q = jp.array(self._mj_model.keyframe("home").qpos) -> We have to add a keyframe called home 
# under wolfgang_scene

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02, # Übernehmbar
      sim_dt=0.002, # Übernehmbar
      episode_length=2000, # Übernehmbar
      action_repeat=1, # Übernehmbar
      action_scale=0.5, # Übernehmbar
      history_len=1, # Übernehmbar
      soft_joint_pos_limit_factor=0.95, # weiche Gelenkpositionsgrenze (95% der Harten) -> sollte übernehmbar sein
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise. -> Übernehmbar (stabilisert das Modell)
          scales=config_dict.create( # Überprüfen, ob die einzelnen Parameter auch auf Wolfgang treffen; Werte erstmal lassen
              hip_pos=0.03,  # rad, Hüfte
              kfe_pos=0.05, # Kniee
              ffe_pos=0.08, # Fußgelenke
              faa_pos=0.03, # Fußgelenke in der Abduktion/Adduktion
              joint_vel=1.5,  # rad/s, Gelenkgeschwindigkeit
              gravity=0.05, # Schwerkraftmessung
              linvel=0.1, # lineare Geschwindigkeit
              gyro=0.2,  # angvel. Winkelgeschwindigkeit
          ),
      ),
      reward_config=config_dict.create( # Sind die Parameter nutzbar?
          scales=config_dict.create(
              # Tracking related rewards.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-1.0,
              base_height=0.0,
              # Energy related rewards.
              torques=-2.5e-5, # Hohe Drehmomente an den Gelenken werden bestraft.
              action_rate=-0.01, # Bestraft schnelle Änderungen in den Aktionen.
              energy=0.0,
              # Feet related rewards.
              feet_clearance=0.0, # Fußfreiheit -> z.B. zu niedrige Schritte
              feet_air_time=2.0, # Belohnt die Zeit, in der die Füße in der Luft sind.
              feet_slip=-0.25, # Bestraft das Rutschen der Füße auf dem Boden
              feet_height=0.0, # Abweichung der Fußhöhe
              feet_phase=1.0, # Belohnt das Einhalten eines gewünschten Gangzyklus.
              # Other rewards.
              stand_still=0.0, # Bewegung ohne Befehle
              alive=0.0, # Lebendig bleiben
              termination=-1.0, # Bestraft das vorzeitige Beenden der Episode
              # Pose related rewards.
              joint_deviation_knee=-0.1, # Bestraft Abweichungen der Kniegelenke von der gewünschten Position.
              joint_deviation_hip=-0.25, # Bestraft Abweichungen der Hüftgelenke von der gewünschten Position.
              dof_pos_limits=-1.0, # Bestrafung der Gelenkpositionsgrenzen
              pose=-1.0, # Bestrafen Abweichungen der gesamten Haltung von einer Zielpose
              ball_velocity=0.3, # 1.0 -> Muss x100 genommen werden
              ball_distance=0,
              robot_ball_orientation = -0.5, # bestes Ergebnis bei 0.085 
              ball_target_deviation = -0.5, #1.0         
          ),
          tracking_sigma=0.5, # Beeinflusst, wie empfindlich die Belohnung auf Abweichungen zwischen der gewünschten Geschwindigkeit (Befehl) und der tatsächlichen Geschwindigkeit des Roboters reagiert.
          max_foot_height=0.1,
          base_height_target=0.5,
      ),
      push_config=config_dict.create( # Konfiguration für zufällige Stöße
          enable=True, # Zufällige Stöße sind aktiviert
          interval_range=[5.0, 10.0], # Zeitbereich zwischen zwei aufeinanderfolgenden Stößen
          magnitude_range=[0.2, 0.5], # Stärke der Stöße
      ),
      lin_vel_x=[0.5, 0.5], # Gewünschte Geschwindigkeiten(x, y, Dreh) 
      lin_vel_y=[-0.5, 0.5],
      ang_vel_yaw=[-0.5, 0.5],
  )


class Joystick(wolfgang_base.WolfgangEnv):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain", # Für Wolfgang entfernen
      config: config_dict.ConfigDict = default_config(), # Übernehmbar
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None, # Übernehmbar
  ):
    super().__init__( # WolfgangEnv wird aufgerufen -> Analysieren
        xml_path=consts.task_to_xml(task).as_posix(), # Wie werden die XML-Dateien verwendet
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()


  def _post_init(self) -> None:
    # XML-Datei überprüfen, ob entsprechende Eigenschaften vorhanden sind, z.B. _mj_model.Eigenschaft
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos) # Keyframe ist ein gespeicherter Zustand des Modells -> Standardposition (Für Wolfgang?)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:25])

    self._ball_body_id = self._mj_model.body("ball").id

    self._target_body_id = self._mj_model.body("target").id

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:-2].T # Erhalten untere und obere Grenze 
    # Berechnen wieche Grenzen -> Übernehmbar
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    hip_indices = []
    hip_joint_names = ["HipYaw", "HipRoll", "HipPitch"] # HR: Hüftrotation, HAA: Hüftadaption/-abduktion
    for side in ["L", "R"]: 
      for joint_name in hip_joint_names:
        hip_indices.append(
            self._mj_model.joint(f"{side}{joint_name}").qposadr - 7
        ) 
    self._hip_indices = jp.array(hip_indices)

    knee_indices = []
    for side in ["L", "R"]:
      knee_indices.append(self._mj_model.joint(f"{side}Knee").qposadr - 7) # Erste sieben Einträge gehören der Basisbewegung an, die Restlichen den Gelenkwinkel
    self._knee_indices = jp.array(knee_indices)

    # Gewichtsvektoren für die Gelenke -> Wie wichtig sind die einzelnen Elemente für die Belohnung?
    # fmt: off
    self._weights = jp.array([
        0, 0, 0,  # right arm
        0, 0, 0,  # left arm
      
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # left leg.
        1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # right leg.

        # 0, 0 # head
    ])
    # fmt: on

    # Speichert IDs und Massen für den Torso und die IMU (Inertial Measurement Unit).
    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id

    # Speichert IDs für die Füße und den Boden.
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id # ID
      sensor_adr = self._mj_model.sensor_adr[sensor_id] # Startindex
      sensor_dim = self._mj_model.sensor_dim[sensor_id] # Dimension
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    # Hinzufügen von Rauschen in den Gelenkstellungen
    qpos_noise_scale = np.zeros(18)
    hip_ids = [0, 1, 2, 6, 7, 8]
    kfe_ids = [3, 9]
    ffe_ids = [4, 10] # joint type...
    faa_ids = [5, 11] # faa is a type of actuator
    qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
    qpos_noise_scale[kfe_ids] = self._config.noise_config.scales.kfe_pos
    qpos_noise_scale[ffe_ids] = self._config.noise_config.scales.ffe_pos
    qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
    self._qpos_noise_scale = jp.array(qpos_noise_scale)

  # Setzen Roboter in einen zufälligen Startzustand
  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q # Gelenkpositionen auf Startwerte setzen
    qvel = jp.zeros(self.mjx_model.nv) # Gelenkgeschwindigkeiten auf 0 setzen

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5) # Zufallszahlen für x und y erzeugt
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy) # Werden auf die Gelenkpositionen addiert
    # Zufällige Drehung um die Z-Achse
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # Zufällige Gelenkpositionen 
    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:25].set(
        qpos[7:25] * jax.random.uniform(key, (18,), minval=0.5, maxval=1.5)
    )

    # Zufällige Gelenkgeschwindigkeiten
    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    # Zufällige Ballposition
    rng, ball_rng = jax.random.split(rng)

    ball_xy = jax.random.uniform(
      ball_rng, (2,), minval = jp.array([-0.5, -0.5]), maxval = jp.array([+0.5, +0.5])
    )

    offset = 1

    ball_xy = jp.where(ball_xy[0] > 0,
                       ball_xy + jp.array([offset, 0]),
                       ball_xy - jp.array([offset, 0]))
    
    ball_xy = jp.where(ball_xy[1] > 0,
                       ball_xy + jp.array([0, offset]),
                       ball_xy - jp.array([0, offset]))

    ball_z = self._mj_model.geom("ball_geom").size[0]
    ball_jid = self._mj_model.joint("ball_freejoint").id
    ball_qposadr = self._mj_model.jnt_qposadr[ball_jid]
    qpos = qpos.at[ball_qposadr:ball_qposadr+3].set(
        jp.concatenate([ball_xy, jp.array([ball_z])])
    )
    qpos = qpos.at[ball_qposadr+3:ball_qposadr+7].set(jp.array([1.0, 0.0, 0.0, 0.0]))

    # Zufällige Targetposition
    # TODO: Restliche qpos-Werte anpassen
    rng, target_rng = jax.random.split(rng)

    target_xy = jax.random.uniform(
      target_rng, (2,), minval = jp.array([-14.0, -14.0]), maxval = jp.array([14.0, 14.0])
    )

    offset = 10.0

    target_xy = jp.where(target_xy[0] > 0,
                         target_xy + jp.array([offset, 0.0]),
                         target_xy - jp.array([offset, 0.0]))

    target_xy = jp.where(target_xy[1] > 0,
                      target_xy + jp.array([0.0, offset]),
                      target_xy - jp.array([0.0, offset]))
    
    target_z = self._mj_model.geom("target_geom").size[1] / 2
    target_jid = self._mj_model.joint("target_joint").id
    target_qposadr = self._mj_model.jnt_qposadr[target_jid]
    qpos = qpos.at[target_qposadr: target_qposadr+3].set(
      jp.concatenate([target_xy, jp.array([target_z])])
    )
    qpos = qpos.at[target_qposadr+3 : target_qposadr+7].set(jp.array([1.0, 0.0, 0.0, 0.0]))

    # Erstellen MuJoCo-Datenobjekt mit den initialien Zuständen
    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:25])

    # Zufällige Gangfrequenz (gait_freq) + Phaseninkrement (phase_dt)
    # Phase, freq=U(1.0, 1.5)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi]) # Startphase für die Füße

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng) # Zufälliger Steuerbefehl

    # Sample push interval. -> Stoßintervall
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    init_dist = jp.linalg.norm(
      data.xpos[self._torso_body_id, :2] - data.xpos[self._ball_body_id, :2]
    )

    # Sollte übernehmbar sein
    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu), # mjx_model.nu -> Anzahl der Aktuatoren
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        "previous_ball_distance": init_dist,
        "time_till_ball_contact_termination": 20 * 1/self.dt, # 30 Sekunden bis zum Ende der Episode, wenn kein Ballkontakt besteht
    }

    # Erstellen Belohnungskomponenten für einzelne Metriken
    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(()) # Maximale Fußhöhe

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id) # Überprüfen, ob Füße den Boden berühren; geom_id -> ID des Fußes, self._floor_geom_id -> ID des Bodens
        for geom_id in self._feet_geom_id
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2) # Belohnung und Ende der Episode auf 0 setzen

    return mjx_env.State(data, obs, reward, done, metrics, info)

  # Simulieren einen Zeitschritt
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    # Stoß -> Übernehmbar
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi) # Stoßrichtung als Winkel
    # Zufällige Stärke des Stoßes
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)]) # Stoßrichtung als x- und y-Komponente
    # Aktivieren Stoß, falls bestimmte Anzahl von Schritten erreicht ist
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable # Aktivieren Stoß, falls entsprechender Konfigurationsparameter aktiviert ist
    # Stoß wird Ausgeführt -> Aktualisieren die Gelenkgeschwindigkeiten
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    # Sollte übernehmbar sein
    motor_targets = self._default_pose + action * self._config.action_scale # Berechnen Zielposition der Gelenke
    # Simulationsschritt ausführen
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    # Übernehmbar -> Berechnen maximale Fußhöhe und Dauer der Füße in der Luft
    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id] # Extrahieren die Positionen der Sites, die den Füßen zugeordnet sind
    p_fz = p_f[..., -1] # Extrahieren die z-Koordinaten

    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    # Überprüfen, ob der Ball berührt wurde
    ball_pos = data.xpos[self._ball_body_id, :2] # Extrahieren die Position des Balls
    torso_pos = data.xpos[self._torso_body_id, :2] # Extrahieren die Position des Torsos
    torso_ball_distance = jp.linalg.norm(torso_pos - ball_pos) # Berechnen die Distanz zwischen Torso und Ball
    first_ball_contact = jp.logical_and(
        state.info["previous_ball_distance"] > torso_ball_distance,
        torso_ball_distance < 0.2, # Überprüfen, ob der Ball berührt wurde
    )

    # Übernehmbar
    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data, state)

    # Übernehmbar? -> _get_reward muss überprüft werden
    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Sollte übernehmbar sein (reward überprüfen) -> Aktualisieren Parameter
    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    state.info["previous_ball_distance"] = torso_ball_distance


    state.info["time_till_ball_contact_termination"] = jp.where(
        first_ball_contact, 
        100000 * 1/self.dt, # Reset Timer, wenn Ball berührt wurde
        state.info["time_till_ball_contact_termination"] - 1
    )

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)

    return state

  # Übernehmbar -> Überpüft, ob Simulation beendet werden soll
  def _get_termination(self, data: mjx.Data, state) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.0 # Überprüfen, ob der Roboter gefallen ist

    no_ball_contact_termination = (state.info["time_till_ball_contact_termination"] <= 0) # Überprüfen, ob der Ballkontakt zu lange her ist

    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any() # | no_ball_contact_termination
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    # Fügen Rauschen hinzu
    # Übernehmbar
    gyro = self.get_gyro(data) # Ausrichtung und Winkelgeschwindigkeit -> Methode in base.py, übernehmbar?
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    # Übernehmbar
    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    # Übernehmbar
    joint_angles = data.qpos[7:25]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._qpos_noise_scale
    )

    # Übernehmbar
    joint_vel = data.qvel[15:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    # Übernehmbar
    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin]) # Überführung der Gangphase in Sinus- und Cosinuswerte (zyklische Darstellung) -> Eindeutigkeit durch Kombination aus Sinus und Cosinus gewährleistet

    # Übernehmbar
    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    # Ball-Statistiken
    torso_pos = data.xpos[self._torso_body_id, :2]
    ball_pos = data.xpos[self._ball_body_id, :2]
    torso_ball_vec = ball_pos - torso_pos

    quat = data.qpos[3:7]
    yaw = self.quat_to_yaw(quat)

    # Relative Ball-Position

    rel_x = torso_ball_vec[0]*jp.cos(yaw) + torso_ball_vec[1]*jp.sin(yaw)
    rel_y = -torso_ball_vec[0]*jp.sin(yaw) + torso_ball_vec[1]*jp.cos(yaw)
    rel_ball_pos_xy = jp.stack([rel_x, rel_y])

    rel_ball_pos_xy_normalized = rel_ball_pos_xy/(jp.linalg.norm(rel_ball_pos_xy) + 1e-6) # L2-Normalisierung

    # Ballgeschwindigkeit

    ball_vel = data.cvel[self._ball_body_id, :2]
    ball_speed = jp.linalg.norm(ball_vel)
    ball_speed = jp.clip(ball_speed, a_min=0.0, a_max = 12.0)

    ball_speed_normalized = ball_speed/12 # 10 wegen max Geschwindigkeit

    reward = jp.square(ball_speed_normalized)

    # Relative Target-Position
    target_pos = data.xpos[self._target_body_id, :2]
    torso_target_vec = target_pos - torso_pos

    rel_target_x = torso_target_vec[0]*jp.cos(yaw) + torso_target_vec[1]*jp.sin(yaw)
    rel_target_y = -torso_target_vec[0]*jp.sin(yaw) + torso_target_vec[1]*jp.cos(yaw)
    rel_target_pos_xy = jp.stack([rel_target_x, rel_target_y])

    rel_target_pos_xy_normalized = rel_target_pos_xy/(jp.linalg.norm(rel_target_pos_xy) + 1e-6) # L2-Normalisierung

    # Aktuelle Zustand des Roboters (als einziger Zustandsvektor gespeichert) -> Übernehmbar
    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 12 Gelenkwindel relativ zur Standardposition
        noisy_joint_vel,  # 12
        info["last_act"],  # 12 Steuerungsbefehle, die im letzten Schritt an die Aktuatoren gesendet wurden
        phase,
        rel_ball_pos_xy_normalized,
        ball_speed_normalized,
        rel_target_pos_xy_normalized
    ])

    accelerometer = self.get_accelerometer(data) # Übernehmbar  -> base.py anpassen
    global_angvel = self.get_global_angvel(data) # Übernehmbar -> base.py anpassen
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel() # Übernehmbar
    root_height = data.qpos[2] # Höhe des Roboters über dem Boden (z-Koordinate) -> Übernehmbar

    # Übernehmbar -> state + glatte Werte + weitere Werte
    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        data.actuator_force,  # 12
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
        torso_pos, # 3
        ball_pos, # 3
        ball_speed,
        target_pos
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward( # Ab hier
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
      
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:24], data.actuator_force),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:25]),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:25], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:25]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:25]),
        "pose": self._cost_pose(data.qpos[7:25]),
        # Ball related rewards      
        "ball_velocity": self._reward_ball_velocity(data),
        "ball_distance": self._reward_ball_distance(data, info),
        "robot_ball_orientation": self._cost_robot_ball_orientation(data),
        "ball_target_deviation": self._cost_ball_target_deviation(data),
    }

  # Tracking rewards.

  # Belohnung für die Geschwindigkeit -> Übernehmbar? -> tracking_sigma richtig gewählt?
  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  # Belohnung für die Winkelgeschwindigkeit -> Übernehmbar? -> tracking_sigma richtig gewählt?
  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards. Bestrafung für unerwünschte Bewegungen -> Übernehmbar? -> Richtige Parameter?

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  # Energy related rewards. -> Bestrafung für hohe Energieverbräuche -> Übernehmbar? -> Richtige Parameter?

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array # qfrc_actuator -> Kräfte, die von den Aktuatoren erzeugt werden
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    c1 = jp.sum(jp.square(act - last_act))
    return c1

  # Other rewards. Übernehmbar? -> Richtige Parameter?

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0) # Negative Werte bleiben erhalten, positive Werte werden auf 0 gesetzt
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None) # Positive Werte bleiben erhalten, negative Werte werden auf 0 gesetzt
    return jp.sum(out_of_limits)

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    return jp.sum(jp.abs(qpos - self._default_pose)) * (cmd_norm < 0.1)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Pose-related rewards.

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    cost = jp.sum(
        jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
    )
    cost *= jp.abs(cmd[1]) > 0.1
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(
        jp.abs(
            qpos[self._knee_indices] - self._default_pose[self._knee_indices]
        )
    )

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    body_vel = self.get_global_linvel(data)[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    del commands  # Unused.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward
  
  # Ball related rewards.

  #TODO: Überprüfen, ob bei minimalen Winkel es den maximalen Reward gibt
  #TODO: Division mit 0 vermeiden
  def _reward_ball_velocity(
      self,
      data: mjx.Data,
  ):
    # Pure ball speed
    max_ball_vel = 12.0 # max ball velocity
    ball_vel = data.cvel[self._ball_body_id, :2]
    ball_speed = jp.linalg.norm(ball_vel)
    ball_speed = jp.clip(ball_speed, a_min=0.0, a_max = max_ball_vel)

    ball_speed_normalized = ball_speed/max_ball_vel 

    # Ball direction
    ball_target_vec = data.xpos[self._target_body_id, :2] - data.xpos[self._ball_body_id, :2]
    cos_angle = jp.where(
      (jp.linalg.norm(ball_vel) > 1e-6) & (jp.linalg.norm(ball_target_vec) > 1e-6),
      jp.clip((ball_vel @ ball_target_vec) / (jp.linalg.norm(ball_vel) * jp.linalg.norm(ball_target_vec)), a_min = -1, a_max = 1),
      -1
    )
    deviation = jp.arccos(cos_angle) # Ergebnis ist immer positiv
    # deviation = (deviation + jp.pi) % jp.pi

    deviation_normalized = (jp.pi - deviation) / jp.pi

    # Reward
    reward = jp.square(ball_speed_normalized * deviation_normalized)

    return reward
  
  def _reward_ball_distance(
      self,
      data: mjx.Data,
      info: dict[str, Any],
  ):
    current_distance = jp.linalg.norm(
      data.xpos[self._torso_body_id, :2] - data.xpos[self._ball_body_id, :2]
    )

    absolute_previous_distance = abs(info['previous_ball_distance'])
    absolute_current_distance = abs(current_distance)

    """
    brutto_reward = jp.where(absolute_previous_distance > absolute_current_distance,
             absolute_previous_distance - absolute_current_distance,
             0)

    brutto_reward = jp.exp(brutto_reward)

    reward = jp.clip(brutto_reward, a_max = 1)
    """

    reward = absolute_previous_distance - absolute_current_distance

    reward = jp.exp(reward)

    """
    reward = jp.where(reward > 0,
             reward**2,
             (-1) * reward**2)
    """

    return reward
  
  def _cost_robot_ball_orientation(
      self,
      data: mjx.Data,
  ):
    base_quat = data.qpos[3:7]

    yaw = self.quat_to_yaw(base_quat)

    # Robot-Ball Angle Diff
    vec_robot_ball = data.xpos[self._ball_body_id, :2] - data.xpos[self._torso_body_id, :2]

    robot_ball_yaw = jp.arctan2(vec_robot_ball[1], vec_robot_ball[0])

    angle_diff_robot_ball = (robot_ball_yaw - yaw + jp.pi) % (2 * jp.pi) - jp.pi 

    cost_robot_ball = (1 - jp.cos(angle_diff_robot_ball))/2

    # Cost Function
    cost = cost_robot_ball

    robot_ball_distance = jp.linalg.norm(
      data.xpos[self._torso_body_id, :2] - data.xpos[self._ball_body_id, :2]
    )

    weight = jp.where(robot_ball_distance > 0.3, 1.0, 0.05)

    return cost * weight
  
  @staticmethod
  def quat_to_yaw(base_quat):
    w = base_quat[0]
    x = base_quat[1]
    y = base_quat[2]
    z = base_quat[3]

    yaw = jp.arctan2(2*(w*z + x*y), 1-2*(y**2 + z**2))

    return yaw

  def _cost_ball_target_deviation(
      self,
      data: mjx.Data,
  ):
    """
    vec_robot_ball = data.xpos[self._ball_body_id, :2] - data.xpos[self._torso_body_id, :2]
    vec_ball_target = data.xpos[self._target_body_id, :2] - data.xpos[self._ball_body_id, :2]
    
    norm_rb = jp.linalg.norm(vec_robot_ball)
    norm_bt = jp.linalg.norm(vec_ball_target)

    cond = jp.where(jp.any(norm_rb <= 1e-6) | jp.any(norm_bt <= 1e-6), True, False)
    reward_true_cond = jp.where(norm_bt <= 1e-6, 1.0, 0.0)
    
    # Neue Methode: atan2
    dot = jp.dot(vec_robot_ball, vec_ball_target)
    cross = vec_robot_ball[0] * vec_ball_target[1] - vec_robot_ball[1] * vec_ball_target[0]
    theta = jp.atan2(cross, dot)  # Signed Winkel
    deviation = jp.abs(theta)  # Unsigned für [0, π]
    deviation_normalized = deviation / jp.pi
    
    reward_false_cond = 1 - deviation_normalized

    reward = jp.where(cond, reward_true_cond, reward_false_cond)

    return reward
    """

    #robot_target hat gute Ergebnisse geliefert -> 0.3 Schranke

    

    base_quat = data.qpos[3:7]

    yaw = self.quat_to_yaw(base_quat)

    # Robot-Ball Angle Diff
    vec_ball_target = data.xpos[self._target_body_id, :2] - data.xpos[self._ball_body_id, :2]

    ball_target_yaw = jp.arctan2(vec_ball_target[1], vec_ball_target[0])

    angle_diff_robot_ball_target_vec = (ball_target_yaw - yaw + jp.pi) % (2 * jp.pi) - jp.pi 

    cost_robot_target = (1 - jp.cos(angle_diff_robot_ball_target_vec))/2

    # Cost Function
    cost = cost_robot_target

    robot_ball_distance = jp.linalg.norm(
      data.xpos[self._torso_body_id, :2] - data.xpos[self._ball_body_id, :2]
    )

    weight = jp.where(robot_ball_distance > 0.3, 0.05, 5.0)

    return cost * weight

    

    """
    base_quat = data.qpos[3:7]

    yaw = self.quat_to_yaw(base_quat)

    # Robot-Ball Angle Diff
    vec_robot_target = data.xpos[self._target_body_id, :2] - data.xpos[self._torso_body_id, :2]

    robot_target_yaw = jp.arctan2(vec_robot_target[1], vec_robot_target[0])

    angle_diff_robot_target = (robot_target_yaw - yaw + jp.pi) % (2 * jp.pi) - jp.pi 

    cost_robot_target = (1 - jp.cos(angle_diff_robot_target))/2

    # Cost Function
    cost = cost_robot_target

    robot_ball_distance = jp.linalg.norm(
      data.xpos[self._torso_body_id, :2] - data.xpos[self._ball_body_id, :2]
    )

    weight = jp.where(robot_ball_distance > 0.3, 0.1, 3.0)

    return cost * weight

    """


  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )
