import os
import random
import time
from dataclasses import dataclass
import io
from PIL import Image
from pathlib import Path
import cv2
import numpy as np

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 10000000
    learning_rate: float = 1e-4
    num_envs: int = 1
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 32
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000
    train_frequency: int = 4
    frame_skip: int = 4
    resolution_width: int = 84
    resolution_height: int = 84
    grayscale: bool = False
    jpeg_quality: int = 50
    use_compression: bool = False
    generate_preview: bool = False  # Added this line

def make_env(env_id, seed, idx, capture_video, run_name, frame_skip, resolution, grayscale, use_compression, jpeg_quality):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        if frame_skip > 0:
            env = MaxAndSkipEnv(env, skip=frame_skip)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, resolution)
        if grayscale:
            env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        if use_compression:
            env = JPEGCompressionWrapper(env, quality=jpeg_quality)

        env.action_space.seed(seed)
        return env

    return thunk

class JPEGCompressionWrapper(gym.Wrapper):
    def __init__(self, env, quality=50):
        super().__init__(env)
        self.quality = quality

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        compressed_observation = self.compress_observation(observation)
        return compressed_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        compressed_observation = self.compress_observation(observation)
        return compressed_observation, info

    def compress_observation(self, observation):
        compressed = []
        for frame in observation:
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=self.quality)
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            compressed.append(np.array(compressed_img))
        return np.array(compressed)

class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Check if the input is grayscale or RGB
        if x.shape[-1] == 3:  # RGB
            x = jnp.transpose(x, (0, 2, 3, 1, 4))
            x = x.reshape((x.shape[0], x.shape[1], x.shape[2], -1))
        else:  # Grayscale
            x = jnp.transpose(x, (0, 2, 3, 1))

        x = x / 255.0
        x = nn.Conv(32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        print(f"Shape after flattening: {x.shape}")
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

class TrainState(TrainState):
    target_params: flax.core.FrozenDict

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def create_preprocessed_video(env, num_frames=100, filename="preprocessed_video.mp4"):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    obs, _ = env.reset()
    
    for _ in range(num_frames):
        if out is None:
            # Determine the shape and color mode
            if len(obs.shape) == 3:  # Grayscale
                height, width = obs.shape[1:]
                is_color = False
                out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height), isColor=False)
            else:  # RGB
                height, width, _ = obs.shape[1:4]
                is_color = True
                out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height), isColor=True)
        
        # If the observation is a stack of frames, take the last one
        frame = obs[-1] if len(obs.shape) > 3 else obs
        
        # Convert grayscale to single channel for video writing
        if not is_color:
            frame = frame[-1]  # Take the last frame in the stack
            frame = frame.squeeze()  # Ensure it's a single channel
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        out.write(frame)
        
        action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
    
    out.release()
    print(f"Preprocessed video saved as {filename}")

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        # Create the experiment tag
        experiment_effects = []
        if args.grayscale:
            experiment_effects.append(f"grayscale")
        if args.use_compression:
            experiment_effects.append(f"jpeg_quality_{args.jpeg_quality}")
        experiment_effects.append(f"resolution_{args.resolution_width}x{args.resolution_height}")
        experiment_effects.append(f"frame_skip_{args.frame_skip}")
        experiment_tag = "_".join(experiment_effects)

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        
        # Add the experiment tag to the wandb config
        wandb.run.tags = wandb.run.tags + (experiment_tag,)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    resolution = (args.resolution_width, args.resolution_height)
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args.frame_skip, resolution, args.grayscale, args.use_compression, args.jpeg_quality) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)
    print(f"Observation shape: {obs.shape}")
    if args.grayscale:
        assert obs.shape == (args.num_envs, 4, args.resolution_height, args.resolution_width), "Unexpected observation shape for grayscale"
    else:
        assert obs.shape == (args.num_envs, 4, args.resolution_height, args.resolution_width, 3), "Unexpected observation shape for RGB"

    q_network = QNetwork(action_dim=envs.single_action_space.n)

    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs),
        target_params=q_network.init(q_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        q_next_target = q_network.apply(q_state.target_params, next_observations)
        q_next_target = jnp.max(q_next_target, axis=-1)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network.apply(q_state.params, obs)
            actions = q_values.argmax(axis=-1)
            actions = jax.device_get(actions)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                loss, old_val, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()

    # Create the preprocessed video only if generate_preview is True
    if args.generate_preview:
        output_dir = Path("preprocessed_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        video_filename = output_dir / f"{args.env_id.replace('/', '_')}_preprocessed.mp4"
        create_preprocessed_video(envs.envs[0], num_frames=100, filename=str(video_filename))
