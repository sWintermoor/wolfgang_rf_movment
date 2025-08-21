import mujoco
import jax
import orbax.checkpoint

from brax.training.agents.ppo.networks import make_inference_fn

# XML
model = mujoco.MjModel.from_xml_path(
    "mujoco_playground/_src/locomotion/wolfgang/xmls/wolfgang_kicking/wolfgang_scene_kicking.xml"
)
data = mujoco.MjData(model)

# Gewichte
agent_path = "mujoco_playground/learning/notebooks/checkpoints/walking_and_strong_kick_model4"
checkpointer = orbax.checkpoint.PyTreeCheckpointer()
agent_params = checkpointer.restore(agent_path)

# Policy-Funktion

jit_inference_fn = jax.jit(make_inference_fn(agent_params, deterministic=True))

def agent(obs):


with mujoco.MjViewer(model, data) as viewer:
    while viewer.is_running():
        action = jit_inference_fn(data.obs)