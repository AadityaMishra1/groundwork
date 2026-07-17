"""Smoke tests for the grasp synthesis pipeline. Fast (<30 s on an M1)."""

import numpy as np
import pytest

from src.grasp_synth.sampler import sample_pregrasp
from src.grasp_synth.scene import CylinderSpec, build_scene, grasp_clearance
from src.grasp_synth.validate import palm_to_wrist, run_trial

import mujoco


@pytest.fixture(scope="module")
def model():
    return build_scene(CylinderSpec())


def test_scene_builds(model):
    assert model.nu == 20            # shadow hand actuators
    assert model.geom("object_geom").id >= 0


def test_clearance_sane(model):
    c = grasp_clearance(model)
    assert 0.01 < c < 0.06, f"clearance {c} outside plausible hand range"


def test_palm_to_wrist_roundtrip(model):
    """Commanded palm pose must be achieved exactly at qpos-set time."""
    rng = np.random.default_rng(0)
    obj = CylinderSpec()
    pg = sample_pregrasp(rng, obj, clearance=grasp_clearance(model))
    pos, quat = palm_to_wrist(model, pg.palm_pos, pg.palm_quat)

    data = mujoco.MjData(model)
    wjnt = model.body("H_rh_forearm").jntadr[0]
    wadr = model.jnt_qposadr[wjnt]
    data.qpos[wadr:wadr + 3] = pos
    data.qpos[wadr + 3:wadr + 7] = quat
    mujoco.mj_forward(model, data)

    palm = data.body("H_rh_palm")
    assert np.linalg.norm(palm.xpos - pg.palm_pos) < 1e-6
    Rp = np.zeros(9)
    mujoco.mju_quat2Mat(Rp, pg.palm_quat)
    assert np.allclose(palm.xmat, Rp, atol=1e-6)


def test_trial_runs_and_is_honest(model):
    """A trial must complete, never weld the object, and report a verdict."""
    rng = np.random.default_rng(1)
    obj = CylinderSpec()
    pg = sample_pregrasp(rng, obj, clearance=grasp_clearance(model))
    r = run_trial(obj, pg.palm_pos, pg.palm_quat, pg.closure, model=model)
    assert r.reason in {"ok", "dropped", "non_hand_contact",
                        "insufficient_fingers", "invalid_start_penetration",
                        "over_squeeze"}
    assert model.neq == 0            # no equality constraints exist at all


def test_object_only_free_joint(model):
    body = model.body("object")
    assert body.jntnum[0] == 1
    jnt = model.joint(body.jntadr[0])
    assert jnt.type[0] == mujoco.mjtJoint.mjJNT_FREE
