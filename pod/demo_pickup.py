"""Choose-and-watch pickup demo: place a cylinder/bottle where YOU want it,
pick its size and mass, and watch one uncut episode of the policy doing the
task. Proof harness for the end-goal chain — runs the grasp policy today
(kneel start), and the same interface carries over to the composed
walk->kneel->grasp->lift policy (--free once the free-root policy exists).

    python -u demo_pickup.py --ckpt <model.pt> --x 0.28 --y -0.18 \
        --radius 0.04 --obj-height 0.20 --mass 0.4 --headless

Object placement is in the robot's frame (robot at origin; +x forward,
-y is the right-hand side). The trained reach corridor is the front-right
arc r 0.20-0.37, bearing -1.05..-0.45 rad — placements outside it are legal
but expect failures (the seated workspace physically ends there).

Protocol limits (docs/EVAL_PROTOCOL.md): radius <= 0.045, mass <= 0.8,
friction fixed at training values. Verdict per episode is the STRICT bar:
object >= 0.40 m held 3 continuous seconds.
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--x", type=float, default=0.28, help="object x, robot frame (m)")
parser.add_argument("--y", type=float, default=-0.18, help="object y, robot frame (m)")
parser.add_argument("--radius", type=float, default=0.045)
# "--height" collides with SimulationApp's window-height int arg
parser.add_argument("--obj-height", type=float, default=0.15)
parser.add_argument("--mass", type=float, default=0.5)
parser.add_argument("--lying", action="store_true", help="spawn on its side")
parser.add_argument("--scan", action="store_true",
                    help="ignore --x/--y: use the env's randomized spawns and "
                         "print each episode's actual coords — find spots that "
                         "succeed, then re-render with --x/--y set to one")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--free", action="store_true",
                    help="free-root robot + gravity (composition physics)")
parser.add_argument("--video_dir", type=str, default="/workspace/demo_videos")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.radius > 0.045 or args.mass > 0.8:
    parser.error("outside protocol limits: radius <= 0.045, mass <= 0.8")

# every episode is a full-task start — no bank assists in a showcase
os.environ["GRASP_FORCE_STAGE"] = "3"
if args.free:
    os.environ["GRASP_FREE"] = "1"

NO_VIDEO = os.environ.get("GRASP_NO_VIDEO", "0") == "1"
args.video = not NO_VIDEO
args.enable_cameras = not NO_VIDEO

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import g1_grasp  # noqa: F401

import cli_args

EP_STEPS = 600  # 12 s at 50 Hz


def main():
    task = "G1-Grasp-v0"
    env_cfg = parse_env_cfg(task, num_envs=1)
    env_cfg.object_cfg.spawn.radius = args.radius
    env_cfg.object_cfg.spawn.height = args.obj_height
    env_cfg.object_cfg.spawn.mass_props.mass = args.mass
    spawn_z = args.radius if args.lying else args.obj_height / 2
    env_cfg.object_cfg.init_state.pos = (args.x, args.y, spawn_z)
    # raised framing: the lift carries the object to ~0.5-0.6 m — a low
    # lookat crops the hold (the money shot) out of the top of the frame
    env_cfg.viewer.eye = (1.5, -1.1, 1.0)
    env_cfg.viewer.lookat = (args.x, args.y, 0.35)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args)

    env = gym.make(task, cfg=env_cfg,
                   render_mode=None if NO_VIDEO else "rgb_array")
    if not NO_VIDEO:
        env = gym.wrappers.RecordVideo(
            env, video_folder=args.video_dir, step_trigger=lambda s: s == 0,
            # must finish inside the run or no file is written
            video_length=args.episodes * EP_STEPS - 50,
            name_prefix="pickup_demo", disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    dev = raw.device

    if args.lying:
        # lying on its side: 90 deg about y, resting at z = radius
        quat = torch.tensor([[math.cos(math.pi / 4), 0.0,
                              math.sin(math.pi / 4), 0.0]], device=dev)
    else:
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    target = torch.tensor([[args.x, args.y, spawn_z]], device=dev)

    def place_object():
        pose = torch.cat([target + raw.scene.env_origins, quat], dim=-1)
        raw.obj.write_root_pose_to_sim(pose)
        raw.obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))

    if not args.scan:
        place_object()

    obs, _ = env.get_observations(), None
    done_count = 0
    got_near = got_contact = got_grasp = got_lift = False
    strict_streak, strict_hit = 0, False
    steps_in_ep = 0
    ep_spawn = (args.x, args.y)
    with torch.inference_mode():
        while done_count < args.episodes and simulation_app.is_running():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            steps_in_ep += 1
            if steps_in_ep == 1:  # object hasn't moved yet: this is the spawn
                sp = raw.obj.data.root_pos_w[0] - raw.scene.env_origins[0]
                ep_spawn = (float(sp[0]), float(sp[1]))
            palm = raw.robot.data.body_pos_w[:, raw.palm_id]
            objw = raw.obj.data.root_pos_w
            objz = float(objw[0, 2] - raw.scene.env_origins[0, 2])
            nf, th = raw._finger_contacts()
            grasped = bool((nf[0] >= 3) & th[0])
            got_near |= float((palm - objw).norm(dim=-1)[0]) < 0.12
            got_contact |= int(nf[0]) >= 1
            got_grasp |= grasped
            got_lift |= objz > 0.22
            strict_streak = strict_streak + 1 if (objz > 0.40 and grasped) else 0
            strict_hit |= strict_streak >= 150
            if bool(dones[0]):
                done_count += 1
                stage = ("SUCCESS(strict)" if strict_hit else
                         "lift_no_hold" if got_lift else
                         "grasp_no_lift" if got_grasp else
                         "contact_no_grasp" if got_contact else
                         "near_no_contact" if got_near else "never_near")
                print(f"PICKUP_EP {done_count}: {stage}  "
                      f"obj=({ep_spawn[0]:.3f},{ep_spawn[1]:.3f}) r={args.radius} "
                      f"h={args.obj_height} m={args.mass} lying={args.lying} "
                      f"len={steps_in_ep}")
                got_near = got_contact = got_grasp = got_lift = False
                strict_streak, strict_hit = 0, False
                steps_in_ep = 0
                if not args.scan:
                    place_object()  # auto-reset randomized it; put it back
    # finalize the recording even when early terminations kept the step
    # count under video_length (os._exit would otherwise lose the file)
    if not NO_VIDEO:
        env.close()


if __name__ == "__main__":
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
