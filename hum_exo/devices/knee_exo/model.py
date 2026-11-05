"""Compose the articulated CAD device with MyoFullBody in SI units."""

from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import mujoco
import musclemimic_models
import numpy as np
from scipy.spatial.transform import Rotation

from hum_exo.devices.knee_exo.mounting import fit_hinges
from hum_exo.paths import project_root, write_portable_xml

ROOT = project_root()
ASSETS = ROOT / "assets/knee_exo"
DEFAULT_MODEL = ROOT / "generated/knee_exo/human_exo.xml"
LINKS = ("base_link", "motor_housing_link", "motor_output_link", "shank_link")
KINDS = ("upper_hinge", "knee", "lower_hinge")
IMUS = ("left_thigh", "right_thigh", "left_shank", "right_shank")
ENCODERS = ("right_knee_encoder_pos", "right_knee_encoder_vel")
MOTORS = ("Exo_R", "Exo_L")
SOURCE_MOTOR_OFFSET_M = 0.045


def numbers(values: np.ndarray) -> str:
    """Serialize numeric arrays without reducing CAD precision."""
    return " ".join(f"{x:.16g}" for x in np.asarray(values).ravel())


def quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return a MuJoCo scalar-first quaternion from a rotation matrix."""
    return Rotation.from_matrix(matrix).as_quat(scalar_first=True)


def transform(data: mujoco.MjData, body: int) -> np.ndarray:
    """Return the world pose of a body [m]."""
    result = np.eye(4)
    result[:3, :3] = data.xmat[body].reshape(3, 3)
    result[:3, 3] = data.xpos[body]
    return result


def import_device(side: str, output: Path) -> tuple[ET.Element, mujoco.MjModel]:
    """Resolve local URDF meshes and retain visual and inertial information."""
    source = ASSETS / f"urdf/knee_exo_hinge_{side.lower()}.urdf"
    root = ET.parse(source).getroot()
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename", "")
        path = ASSETS / filename.split("/", 3)[3] if filename.startswith("package://") else source.parent / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", str(path.resolve()))
    extension = root.find("mujoco")
    if extension is None:
        extension = ET.SubElement(root, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.update(discardvisual="false", fusestatic="false", strippath="false")
    urdf = output / f"device_{side}.urdf"
    ET.ElementTree(root).write(urdf)
    model = mujoco.MjModel.from_xml_path(str(urdf))
    xml = output / f"device_{side}.xml"
    mujoco.mj_saveLastXML(str(xml), model)
    return ET.parse(xml).getroot(), model


def build_model(output: Path = DEFAULT_MODEL, torque_nm: float = 20.0, fitting_profile: dict | None = None) -> Path:
    """Build a fullbody model with two torque motors and six device joints.

    Motor command [-1, 1] maps to [-torque_nm, torque_nm] N m. Cuff welds
    model numerical attachment; their parameters do not describe measured skin.
    """
    if not np.isfinite(torque_nm) or torque_nm <= 0:
        raise ValueError("torque_nm must be positive and finite")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = musclemimic_models.get_xml_path("myofullbody")
    spec = mujoco.MjSpec.from_file(str(source))
    spec.meshdir = str((Path(spec.modelfiledir) / spec.meshdir).resolve())
    spec.texturedir = str((Path(spec.modelfiledir) / spec.texturedir).resolve())
    human = spec.compile()
    hd = mujoco.MjData(human)
    mujoco.mj_forward(human, hd)
    root = ET.fromstring(spec.to_xml())
    # New joint insertion changes qpos ordering and invalidates source keyframes.
    if root.find("keyframe") is not None:
        root.remove(root.find("keyframe"))
    mounting = json.loads((ASSETS / "source_mounting.json").read_text())
    fitting = json.loads((ASSETS / "mounting.json").read_text())
    if fitting_profile is None and fitting.get("fitting_profile"):
        profile_path = (ASSETS / fitting["fitting_profile"]).resolve()
        if not profile_path.is_relative_to(ASSETS.resolve()):
            raise ValueError("The fitting profile must be inside this device asset package")
        fitting_profile = json.loads(profile_path.read_text())
    if fitting_profile is not None:
        if fitting_profile.get("schema_version") != 1 or fitting_profile.get("device_id") != "knee_exo":
            raise ValueError("Unsupported knee fitting profile")
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ASSETS / "urdf").glob("*.urdf")}
        if fitting_profile.get("urdf_sha256") != checksums:
            raise ValueError("Fitting profile URDF versions differ from the current assets")
    interface = json.loads((ASSETS / "training_interface.json").read_text())
    dynamics = interface["dynamics"]
    asset = root.find("asset")
    equality, contact, sensors, actuators = (root.find(k) for k in ("equality", "contact", "sensor", "actuator"))
    if contact is None:
        contact = ET.SubElement(root, "contact")
    if sensors is None:
        sensors = ET.SubElement(root, "sensor")
    alignment = {}
    for side in ("R", "L"):
        imported, dm = import_device(side, output.parent)
        dd = mujoco.MjData(dm)
        mujoco.mj_forward(dm, dd)
        mount = mounting["sides"][side]
        fitted = np.eye(4)
        fitted[:3, :3] = Rotation.from_euler("xyz", mount["rotation_rpy_deg"], degrees=True).as_matrix()
        fitted[:3, 3] = mount["translation_m"]
        housing_zero = transform(dd, dm.body("motor_housing_link").id)
        for kind in ("upper_hinge", "lower_hinge"):
            dd.qpos[dm.joint(kind + "_joint").qposadr[0]] = np.deg2rad(mount[kind + "_deg"])
        mujoco.mj_forward(dm, dd)
        fitted = fitted @ housing_zero @ np.linalg.inv(transform(dd, dm.body("motor_housing_link").id))
        femur = human.body("femur_" + side.lower()).id
        human_knee = human.joint("knee_angle_" + side.lower()).id
        femur_pose = transform(hd, femur)
        human_axis = femur_pose[:3, :3].T @ hd.xaxis[human_knee]
        human_pivot = (np.linalg.inv(femur_pose) @ np.r_[hd.xanchor[human_knee], 1])[:3]
        device_knee = dm.joint("knee_joint").id
        device_axis = fitted[:3, :3] @ dd.xaxis[device_knee]
        # Minimal rotation preserves mounting orientation while matching the new knee axis.
        cross = np.cross(device_axis, human_axis)
        angle = np.arctan2(np.linalg.norm(cross), np.dot(device_axis, human_axis))
        if np.linalg.norm(cross) < 1e-10 and np.dot(device_axis, human_axis) < 0:
            raise ValueError("Antiparallel installation requires explicit calibration")
        correction = Rotation.from_rotvec(cross / max(np.linalg.norm(cross), 1e-10) * angle).as_matrix()
        old_pivot = fitted[:3, :3] @ dd.xanchor[device_knee] + fitted[:3, 3]
        lateral_offset = np.copysign(fitting["motor_axis_offset_m"], np.dot(old_pivot - human_pivot, human_axis))
        fitted[:3, :3] = correction @ fitted[:3, :3]
        fitted[:3, 3] = human_pivot + lateral_offset * human_axis - fitted[:3, :3] @ dd.xanchor[device_knee]
        if fitting_profile is None:
            fitted, cuff_fit = fit_hinges(
                human,
                hd,
                dm,
                dd,
                femur_pose,
                fitted,
                fitting["bone_clearance_m"],
                upper_hinge_deg=fitting.get("upper_hinge_deg"),
                upper_visual_gap_m=fitting.get("upper_visual_gap_m", 0.0),
            )
        else:
            pose = fitting_profile["sides"][side]
            for kind in KINDS:
                joint = dm.joint(kind + "_joint")
                value = np.deg2rad(pose["joint_angles_deg"][kind])
                if not np.isfinite(value) or not joint.range[0] <= value <= joint.range[1]:
                    raise ValueError(f"Profile {side} {kind} exceeds the CAD joint range")
                dd.qpos[joint.qposadr[0]] = value
            mujoco.mj_forward(dm, dd)
            fitted = np.eye(4)
            fitted[:3, :3] = Rotation.from_quat(pose["mount_quat"], scalar_first=True).as_matrix()
            fitted[:3, 3] = pose["mount_pos_m"]
            if not np.isfinite(fitted).all():
                raise ValueError("Nonfinite fitting profile pose")
            cuff_fit = {"source": "interactive fitting profile", "joint_angles_deg": pose["joint_angles_deg"]}
            device_axis = fitted[:3, :3] @ dd.xaxis[device_knee]
            angle = np.arccos(np.clip(np.dot(device_axis, human_axis), -1, 1))
            lateral_offset = float(
                np.dot(fitted[:3, :3] @ dd.xanchor[device_knee] + fitted[:3, 3] - human_pivot, human_axis)
            )
        previous = femur_pose
        names = []
        for mesh in imported.findall("asset/mesh"):
            item = copy.deepcopy(mesh)
            item.set("name", f"knee_{side}_{item.get('name')}")
            asset.append(item)
        for i, link in enumerate(LINKS):
            source_body = imported.find(f".//body[@name='{link}']")
            name = f"hinge_{side}_{link}"
            names.append(name)
            pose = femur_pose @ fitted @ transform(dd, dm.body(link).id)
            relative = np.linalg.inv(previous) @ pose
            parent = root.find(f".//body[@name='{names[i - 1] if i else 'femur_' + side.lower()}']")
            body = ET.SubElement(
                parent, "body", name=name, pos=numbers(relative[:3, 3]), quat=numbers(quaternion(relative[:3, :3]))
            )
            body.append(copy.deepcopy(source_body.find("inertial")))
            if i:
                kind = KINDS[i - 1]
                joint = copy.deepcopy(source_body.find("joint"))
                joint.set("name", f"exo_{kind}_{side.lower()}")
                joint.set("ref", str(dd.qpos[i - 1]))
                joint.set("damping", str(dynamics["passive_hinge_damping"]))
                joint.set("armature", str(dynamics["joint_armature"]))
                body.append(joint)
            for geom in source_body.findall("geom[@group='1']"):
                item = copy.deepcopy(geom)
                item.attrib.update(
                    name=f"{name}_visual",
                    mesh=f"knee_{side}_{item.get('mesh')}",
                    contype="1",
                    conaffinity="1",
                    group="0",
                    rgba=numbers(mount["rgba"]),
                )
                body.append(item)
            previous = pose
        tibia = "tibia_" + side.lower()
        relative = np.linalg.inv(transform(hd, human.body(tibia).id)) @ previous
        ET.SubElement(
            equality,
            "weld",
            name=f"knee_cuff_{side.lower()}",
            body1=tibia,
            body2=names[-1],
            relpose=numbers(np.r_[relative[:3, 3], quaternion(relative[:3, :3])]),
            solref=numbers(dynamics["strap_weld_solref"]),
            solimp=numbers(dynamics["strap_weld_solimp"]),
            torquescale=str(dynamics["strap_weld_torquescale"]),
        )
        for first, second in combinations(names, 2):
            ET.SubElement(contact, "exclude", body1=first, body2=second)
        ET.SubElement(contact, "exclude", body1=tibia, body2=names[-1])
        # The fullbody's coarse leg capsules and convex cuff hulls overlap at
        # a bone-calibrated mount. Attachment constraints carry these internal
        # loads; retain ground, opposite leg and other external collisions.
        for bone in ("femur_", "tibia_", "patella_"):
            for device_body in names:
                if bone == "tibia_" and device_body == names[-1]:
                    continue
                ET.SubElement(contact, "exclude", body1=bone + side.lower(), body2=device_body)
        ET.SubElement(
            actuators,
            "motor",
            name=f"Exo_{side}",
            joint=f"exo_knee_{side.lower()}",
            gear=str(torque_nm),
            ctrllimited="true",
            ctrlrange="-1 1",
            forcelimited="true",
            forcerange="-1 1",
        )
        alignment[side] = {
            "mount_pos_m": fitted[:3, 3].tolist(),
            "mount_quat": quaternion(fitted[:3, :3]).tolist(),
            "axis_correction_deg": float(np.rad2deg(angle)),
            "motor_axis_offset_m": float(lateral_offset),
            "cuff_fit": cuff_fit,
        }
    imu = json.loads((ASSETS / "imu_mounts.json").read_text())
    for entry in imu["sensors"]:
        body = root.find(f".//body[@name='{entry['body']}']")
        name = entry["name"]
        ET.SubElement(
            body,
            "site",
            name=f"imu_{name}",
            pos=numbers(entry["pos_m"]),
            size=".005",
            group="3",
            quat=numbers(Rotation.from_euler("xyz", entry["rpy_deg"], degrees=True).as_quat(scalar_first=True)),
        )
        ET.SubElement(sensors, "accelerometer", name=f"imu_{name}_acc", site=f"imu_{name}")
        ET.SubElement(sensors, "gyro", name=f"imu_{name}_gyro", site=f"imu_{name}")
    for kind, name in zip(("jointpos", "jointvel"), ENCODERS, strict=True):
        ET.SubElement(sensors, kind, name=name, joint="exo_knee_r")
    root.set("model", "MyoFullBody_KneeExo")
    write_portable_xml(root, output)
    combined = mujoco.MjModel.from_xml_path(str(output))
    report = {
        "source_model_package": str(source),
        "torque_nm": torque_nm,
        "nq": combined.nq,
        "nv": combined.nv,
        "nu": combined.nu,
        "device_mass_kg": float(combined.body_mass.sum() - human.body_mass.sum()),
        "extra_lateral_offset_m": dict.fromkeys(("R", "L"), fitting["motor_axis_offset_m"] - SOURCE_MOTOR_OFFSET_M),
        "fitting": fitting,
        "fitting_profile": fitting_profile,
        "alignment": alignment,
        "source_sha256": {
            str(p.relative_to(ASSETS)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ASSETS / "urdf").glob("*.urdf")
        },
        "coupling": "fixed thigh mount and numerical tibia cuff weld; ideal torque motors",
        "calibration_status": "installation from interactive profile; inspect its audit and validate dynamic fit"
        if fitting_profile
        else "neutral axes aligned; upper/hand muscle checks and lower bone clearance fitted; dynamic fit requires validation",
        "collision_exclusions": "same-device pairs and same-side femur/tibia/patella against mounted CAD; external collisions retained",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return output
