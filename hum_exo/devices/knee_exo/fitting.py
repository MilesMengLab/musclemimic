"""Kinematic installation editing and contact-point fitting for the knee CAD."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from hum_exo.devices.knee_exo.model import KINDS, LINKS, build_model, quaternion, transform
from hum_exo.devices.knee_exo.mounting import nearest_bone
from hum_exo.fitting import FitParameter
from hum_exo.geometry import capsule_surface_gaps, mesh_geometry, rendered_tendon_capsules
from hum_exo.paths import resolve_model
from hum_exo.preview import alignment

POINT_QUERY_BATCH = 256
CONTACT_BANDS = ((0.50, 0.78), (0.20, 0.43), (0.73, 1.0), (0.0, 0.27))
CONTACT_LABELS = ("大腿上端", "大腿下端", "小腿近膝端", "小腿远端")
AXIS_DISTANCE_TOLERANCE_MM = 0.05
AXIS_ANGLE_TOLERANCE_DEG = 0.01
MUSCLE_OVERLAP_TOLERANCE_M = 0.0005
SOLVE_MAX_EVALUATIONS = 35


def _versioned_path(path: Path, stamp: str) -> Path:
    """Keep each profile/model snapshot, including simultaneous clock values."""
    stem = re.sub(r"_\d{8}-\d{6}-\d{6}(?:-\d+)?$", "", path.stem)
    counter = 0
    while True:
        extra = f"-{counter:02d}" if counter else ""
        candidate = path.with_name(f"{stem}_{stamp}{extra}{path.suffix}")
        artifacts = (
            candidate,
            candidate.with_name(candidate.stem + "_model.xml"),
            candidate.with_name(candidate.stem + "_model.json"),
        )
        if not any(p.exists() for p in artifacts):
            return candidate
        counter += 1


def point_capsule_gaps(points: np.ndarray, segments: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed point-to-rendered-capsule gaps [m] and nearest surface points."""
    points = np.atleast_2d(points)
    if not len(segments):
        raise ValueError("No muscle visualization capsules available for this leg")
    edge = segments[:, 1] - segments[:, 0]
    norm = np.sum(edge * edge, axis=1)
    gaps, closest = [], []
    for start in range(0, len(points), POINT_QUERY_BATCH):
        batch = points[start : start + POINT_QUERY_BATCH]
        delta = batch[:, None] - segments[None, :, 0]
        t = np.sum(delta * edge, axis=2) / np.maximum(norm, 1e-24)
        axis_point = segments[None, :, 0] + np.clip(t, 0, 1)[..., None] * edge
        offsets = batch[:, None] - axis_point
        lengths = np.linalg.norm(offsets, axis=2)
        signed = lengths - radii
        ids = np.argmin(signed, axis=1)
        rows = np.arange(len(batch))
        gaps.extend(signed[rows, ids])
        closest.extend(
            axis_point[rows, ids] + offsets[rows, ids] / np.maximum(lengths[rows, ids, None], 1e-12) * radii[ids, None]
        )
    return np.asarray(gaps), np.asarray(closest)


class KneeFittingSession:
    """Edit complete articulated chains, keeping each motor housing as the root."""

    sides = ("R", "L")
    contact_labels = CONTACT_LABELS
    controls = (
        FitParameter("offset_mm", "电机轴向偏置 (mm)", 20, 90, 0.1),
        FitParameter("upper_deg", "上合页 (°)", -20, 20, 0.1),
        FitParameter("lower_deg", "下合页 (°)", -20, 20, 0.1),
        FitParameter("twist_deg", "绕膝轴旋转装配体 (°)", -15, 15, 0.1),
        FitParameter("forward_mm", "前后平移 (mm)", -30, 30, 0.1, True),
        FitParameter("vertical_mm", "上下平移 (mm)", -30, 30, 0.1, True),
        FitParameter("tilt_forward_deg", "绕前后方向倾斜 (°)", -15, 15, 0.1, True),
        FitParameter("tilt_vertical_deg", "绕上下方向倾斜 (°)", -15, 15, 0.1, True),
    )

    def __init__(self, adapter, model_path: Path | None = None) -> None:
        self.adapter = adapter
        self.source_model = model_path or adapter.build_model()
        self.model = mujoco.MjModel.from_xml_path(str(self.source_model))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.original_pos, self.original_quat = self.model.body_pos.copy(), self.model.body_quat.copy()
        self.initial_qpos = self.model.qpos0.copy()
        self.state, self.baseline, self.contacts = {}, {}, {}
        self.last_audit = None
        self.revision = 0
        self._audit_revision = -1
        self.leg_capsules = {}
        for side in self.sides:
            suffix = side.lower()
            human = self.model.joint("knee_angle_" + suffix).id
            motor = self.model.joint("exo_knee_" + suffix).id
            housing = self.model.body(f"hinge_{side}_motor_housing_link").id
            axis = self.data.xaxis[human].copy()
            pivot = self.data.xanchor[human].copy()
            distance = float((self.data.xanchor[motor] - pivot) @ axis)
            forward = np.array([0.0, 1.0, 0.0])
            forward -= axis * (forward @ axis)
            forward /= np.linalg.norm(forward)
            vertical = np.cross(axis, forward)
            self.baseline[side] = {
                "housing": transform(self.data, housing),
                "axis": axis,
                "pivot": pivot,
                "offset": abs(distance),
                "sign": np.sign(distance),
                "forward": forward,
                "vertical": vertical,
            }
            self.state[side] = dict.fromkeys((c.key for c in self.controls), 0.0)
            self.state[side].update(
                offset_mm=abs(distance) * 1000,
                upper_deg=float(np.rad2deg(self.data.joint("exo_upper_hinge_" + suffix).qpos[0])),
                lower_deg=float(np.rad2deg(self.data.joint("exo_lower_hinge_" + suffix).qpos[0])),
            )
            self.leg_capsules[side] = rendered_tendon_capsules(self.model, self.data, body_roots=("femur_" + suffix,))
            self.contacts[side] = self._initial_contacts(side)
        report_path = self.source_model.with_suffix(".json")
        if report_path.is_file():
            saved = json.loads(report_path.read_text()).get("fitting_profile")
            if saved:
                for side in self.sides:
                    for index, contact in enumerate(saved["sides"][side].get("contacts", [])):
                        if index < len(self.contacts[side]) and contact["link"] == self.contacts[side][index]["link"]:
                            self.contacts[side][index]["local_point"] = contact["local_point"]
        self.initial_state = copy.deepcopy(self.state)
        self.initial_contacts = copy.deepcopy(self.contacts)

    def parameters(self, side: str) -> dict[str, float]:
        return self.state[side].copy()

    def _apply(self) -> None:
        self.model.body_pos[:] = self.original_pos
        self.model.body_quat[:] = self.original_quat
        self.data.qpos[:] = self.initial_qpos
        for side, values in self.state.items():
            self.data.joint("exo_upper_hinge_" + side.lower()).qpos[0] = np.deg2rad(values["upper_deg"])
            self.data.joint("exo_lower_hinge_" + side.lower()).qpos[0] = np.deg2rad(values["lower_deg"])
        mujoco.mj_kinematics(self.model, self.data)
        relative_housing = {}
        for side in self.sides:
            base = self.model.body(f"hinge_{side}_base_link").id
            housing = self.model.body(f"hinge_{side}_motor_housing_link").id
            relative_housing[side] = np.linalg.inv(transform(self.data, base)) @ transform(self.data, housing)
        for side, values in self.state.items():
            b = self.baseline[side]
            twist = Rotation.from_rotvec(b["axis"] * np.deg2rad(values["twist_deg"])).as_matrix()
            tilt = Rotation.from_rotvec(b["forward"] * np.deg2rad(values["tilt_forward_deg"])).as_matrix()
            tilt = Rotation.from_rotvec(b["vertical"] * np.deg2rad(values["tilt_vertical_deg"])).as_matrix() @ tilt
            rotation = tilt @ twist
            target = b["housing"].copy()
            target[:3, :3] = rotation @ target[:3, :3]
            pivot = b["pivot"] + b["sign"] * b["offset"] * b["axis"]
            target[:3, 3] = pivot + rotation @ (target[:3, 3] - pivot)
            target[:3, 3] += b["sign"] * (values["offset_mm"] / 1000 - b["offset"]) * b["axis"]
            target[:3, 3] += values["forward_mm"] / 1000 * b["forward"] + values["vertical_mm"] / 1000 * b["vertical"]
            femur = self.model.body("femur_" + side.lower()).id
            local = np.linalg.inv(transform(self.data, femur)) @ target @ np.linalg.inv(relative_housing[side])
            base = self.model.body(f"hinge_{side}_base_link").id
            self.model.body_pos[base] = local[:3, 3]
            self.model.body_quat[base] = quaternion(local[:3, :3])
        mujoco.mj_forward(self.model, self.data)

    def update(self, side: str, values: dict[str, float], mirror: bool = True, lock_axes: bool = True) -> None:
        if side not in self.sides:
            raise ValueError("Choose R or L")
        parameters = self.state[side] | values
        for control in self.controls:
            value = parameters[control.key]
            if not np.isfinite(value) or not control.minimum <= value <= control.maximum:
                raise ValueError(f"{control.label} is outside its supported range")
            if lock_axes and control.advanced:
                parameters[control.key] = 0.0
        self.state[side] = parameters
        if mirror:
            self.state["L" if side == "R" else "R"] = parameters.copy()
        self._apply()
        self.revision += 1

    def reset(self) -> None:
        self.state = copy.deepcopy(self.initial_state)
        self.contacts = copy.deepcopy(self.initial_contacts)
        self._apply()
        self.revision += 1

    def _initial_contacts(self, side: str) -> list[dict]:
        result = []
        for index, band in enumerate(CONTACT_BANDS):
            link = "base_link" if index < 2 else "shank_link"
            body = self.model.body(f"hinge_{side}_{link}").id
            geom = self.model.geom(f"hinge_{side}_{link}_visual").id
            vertices, faces = mesh_geometry(self.model, self.data, geom)
            lo, hi = vertices[:, 2].min(), vertices[:, 2].max()
            selected = (vertices[:, 2] >= lo + band[0] * (hi - lo)) & (vertices[:, 2] <= lo + band[1] * (hi - lo))
            triangles = vertices[faces]
            centers = triangles.mean(1)
            triangle_band = (centers[:, 2] >= lo + band[0] * (hi - lo)) & (centers[:, 2] <= lo + band[1] * (hi - lo))
            candidates = np.r_[
                vertices[selected],
                centers[triangle_band],
                ((triangles[:, 0] + triangles[:, 1]) / 2)[triangle_band],
                ((triangles[:, 1] + triangles[:, 2]) / 2)[triangle_band],
                ((triangles[:, 2] + triangles[:, 0]) / 2)[triangle_band],
            ]
            ids = np.arange(len(candidates))
            if not len(ids):
                raise ValueError("Empty fitting contact region")
            gaps, _ = point_capsule_gaps(candidates, *self.leg_capsules[side])
            point = candidates[np.argmin(gaps)]
            local = (point - self.data.xpos[body]) @ self.data.xmat[body].reshape(3, 3)
            result.append({"label": CONTACT_LABELS[index], "body": body, "link": link, "local_point": local.tolist()})
        return result

    def contact_readings(self, side: str) -> list[dict]:
        readings = []
        for point in self.contacts[side]:
            body = point["body"]
            world = self.data.xpos[body] + self.data.xmat[body].reshape(3, 3) @ point["local_point"]
            gaps, nearest = point_capsule_gaps(world, *self.leg_capsules[side])
            readings.append(
                {
                    "label": point["label"],
                    "position": world.tolist(),
                    "nearest": nearest[0].tolist(),
                    "gap_mm": float(gaps[0] * 1000),
                }
            )
        return readings

    def pick_contact(self, side: str, index: int, geom_id: int, origin: np.ndarray, direction: np.ndarray) -> bool:
        point = self.contacts[side][index]
        if int(self.model.geom_bodyid[geom_id]) != point["body"]:
            return False
        vertices, faces = mesh_geometry(self.model, self.data, geom_id)
        triangles = vertices[faces]
        edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        h = np.cross(direction, edge2)
        det = np.sum(edge1 * h, axis=1)
        inv = np.divide(1, det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
        delta = origin - triangles[:, 0]
        u = inv * np.sum(delta * h, axis=1)
        q = np.cross(delta, edge1)
        v = inv * (q @ direction)
        t = inv * np.sum(edge2 * q, axis=1)
        valid = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        if not valid.any():
            return False
        world = origin + direction * t[valid].min()
        body = point["body"]
        point["local_point"] = ((world - self.data.xpos[body]) @ self.data.xmat[body].reshape(3, 3)).tolist()
        self.revision += 1
        return True

    def solve_contacts(
        self, side: str, targets_mm: list[float | None], mirror: bool = True, lock_axes: bool = True
    ) -> dict:
        if len(targets_mm) != len(self.contact_labels):
            raise ValueError("Provide one target per contact point")
        selected = [i for i, target in enumerate(targets_mm) if target is not None]
        if not selected:
            raise ValueError("Select at least one contact point")
        controls = [control for control in self.controls if not lock_axes or not control.advanced]
        keys = [control.key for control in controls]
        start = np.array([self.state[side][key] for key in keys])
        initial = self.state[side].copy()
        targets = np.array([targets_mm[i] for i in selected], dtype=float)
        if not np.isfinite(targets).all():
            raise ValueError("Contact targets must be finite")
        scales = np.array([5 if key.endswith("mm") else 3 for key in keys], dtype=float)

        def residual(x):
            self.state[side] = initial | dict(zip(keys, x, strict=True))
            self._apply()
            gaps = np.array([r["gap_mm"] for r in self.contact_readings(side)])
            return np.r_[gaps[selected] - targets, 0.1 * (x - start) / scales]

        try:
            solved = least_squares(
                residual,
                start,
                bounds=([c.minimum for c in controls], [c.maximum for c in controls]),
                max_nfev=SOLVE_MAX_EVALUATIONS,
                diff_step=1e-3,
            )
            values = dict(zip(keys, solved.x, strict=True))
            self.update(side, values, mirror=mirror, lock_axes=lock_axes)
        except Exception:
            self.state[side] = initial
            self._apply()
            raise
        gaps = np.array([r["gap_mm"] for r in self.contact_readings(side)])
        return {
            "success": bool(solved.success),
            "evaluations": solved.nfev,
            "residual_mm": (gaps[selected] - targets).tolist(),
            "achieved_gap_mm": gaps.tolist(),
            "message": "Each cuff is rigid; several requested gaps may conflict. Run full interference check before applying.",
        }

    def audit(self) -> dict:
        bones = [
            (self.model.geom(i).name, mesh_geometry(self.model, self.data, i))
            for i in range(self.model.ngeom)
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
            and self.model.geom_group[i] == 0
            and not self.model.geom(i).name.startswith("hinge_")
        ]
        muscles = rendered_tendon_capsules(self.model, self.data)
        hands = rendered_tendon_capsules(self.model, self.data, body_roots=("ulna_r", "ulna_l"))
        report = {
            "alignment": alignment(self.model, self.data, self.adapter),
            "alignment_tolerance": {"distance_mm": AXIS_DISTANCE_TOLERANCE_MM, "angle_deg": AXIS_ANGLE_TOLERANCE_DEG},
            "sides": {},
            "failures": [],
            "warnings": [],
        }
        valid = True
        for name, metric in report["alignment"].items():
            aligned = (
                metric["axis_line_distance_mm"] <= AXIS_DISTANCE_TOLERANCE_MM
                and metric["axis_angle_deg"] <= AXIS_ANGLE_TOLERANCE_DEG
            )
            valid &= aligned
            if not aligned:
                report["failures"].append(
                    f"{name}: 膝轴未共轴, 轴距 {metric['axis_line_distance_mm']:.4f} mm, "
                    f"夹角 {metric['axis_angle_deg']:.4f}°"
                )
        for side in self.sides:
            report["sides"][side] = {}
            for link in LINKS:
                geom = self.model.geom(f"hinge_{side}_{link}_visual").id
                surface = mesh_geometry(self.model, self.data, geom)
                bone = nearest_bone(surface, bones)
                report["sides"][side][link] = {
                    "bone_gap_mm": bone["distance_m"] * 1000,
                    "bone": bone["bone"],
                    "bone_crossings": bone["crossing_edges"],
                }
                if bone["crossing_edges"] or bone["distance_m"] <= 0:
                    report["warnings"].append(f"{side} {link}: 与骨骼 {bone['bone']} 相交")
                if link == "base_link":
                    hand_gaps = capsule_surface_gaps(surface, *hands, 0.005)
                    muscle_gaps = capsule_surface_gaps(surface, *muscles, 0.005)
                    report["sides"][side][link].update(
                        hand_gap_mm=float(hand_gaps.min() * 1000), muscle_gap_mm=float(muscle_gaps.min() * 1000)
                    )
                    if hand_gaps.min() < 0:
                        report["warnings"].append(f"{side} 大腿绑缚: 与手部显示几何相交")
                    if muscle_gaps.min() < -MUSCLE_OVERLAP_TOLERANCE_M:
                        report["warnings"].append(
                            f"{side} 大腿绑缚: 肌肉显示几何重叠 {-muscle_gaps.min() * 1000:.3f} mm, "
                            f"超过界面阈值 {MUSCLE_OVERLAP_TOLERANCE_M * 1000:.3f} mm"
                        )
        report["applicable"] = bool(valid)
        report["scope"] = (
            "Neutral CAD and rendered muscle geometry; numerical attachment, not skin pressure or dynamic validation."
        )
        self.last_audit, self._audit_revision = report, self.revision
        return report

    def profile(self) -> dict:
        result = {
            "schema_version": 1,
            "device_id": "knee_exo",
            "sides": {},
            "urdf_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (self.adapter.spec.asset_root / "urdf").glob("*.urdf")
            },
        }
        for side in self.sides:
            base = self.model.body(f"hinge_{side}_base_link").id
            pose = self.model.body_quat[base]
            result["sides"][side] = {
                "mount_pos_m": self.model.body_pos[base].tolist(),
                "mount_quat": pose.tolist(),
                "joint_angles_deg": {
                    kind: float(np.rad2deg(self.data.joint(f"exo_{kind}_{side.lower()}").qpos[0])) for kind in KINDS
                },
                "controls": self.parameters(side),
                "contacts": self.contacts[side],
            }
        if self._audit_revision == self.revision:
            result["audit"] = self.last_audit
        return result

    def is_applied(self) -> bool:
        """Check physical mounting against the enabled profile, ignoring UI coordinates."""
        try:
            asset = self.adapter.spec.asset_root
            config = json.loads((asset / "mounting.json").read_text())
            if not config.get("fitting_profile"):
                return False
            path = (asset / config["fitting_profile"]).resolve()
            if not path.is_relative_to(asset.resolve()):
                return False
            installed = json.loads(path.read_text())
            current = self.profile()
            if installed.get("urdf_sha256") != current["urdf_sha256"]:
                return False
            for side in self.sides:
                active, pose = installed["sides"][side], current["sides"][side]
                if not np.allclose(active["mount_pos_m"], pose["mount_pos_m"], atol=1e-9, rtol=0):
                    return False
                quat, other = np.asarray(active["mount_quat"]), np.asarray(pose["mount_quat"])
                if not (np.allclose(quat, other, atol=1e-9, rtol=0) or np.allclose(quat, -other, atol=1e-9, rtol=0)):
                    return False
                for kind in KINDS:
                    if not np.isclose(
                        active["joint_angles_deg"][kind], pose["joint_angles_deg"][kind], atol=1e-7, rtol=0
                    ):
                        return False
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def save(self, output: Path, apply: bool = False) -> Path:
        """Save a new version after checking axes; retain interference as warnings."""
        output = Path(output)
        if output.suffix.lower() != ".json":
            raise ValueError("Save a .json fitting profile")
        if output.resolve().is_relative_to(self.adapter.spec.asset_root.resolve()):
            raise ValueError("Save drafts outside device assets; use Apply to update the installation")
        report = self.audit()
        if not report["applicable"]:
            raise ValueError("膝轴未对齐。不能生成模型或应用安装。\n" + "\n".join(report["failures"]))
        saved_at = datetime.now().astimezone()
        stamp = saved_at.strftime("%Y%m%d-%H%M%S-%f")
        output = _versioned_path(output, stamp)
        output.parent.mkdir(parents=True, exist_ok=True)
        profile = self.profile()
        profile["audit"] = report
        profile["saved_at"] = saved_at.isoformat()
        # Model building writes a JSON report beside its XML. Keep that report
        # separate from the editable installation profile.
        model_output = output.with_name(output.stem + "_model.xml")
        build_model(model_output, fitting_profile=profile)
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, delete=False, suffix=".json") as stream:
            json.dump(profile, stream, indent=2)
            temporary = Path(stream.name)
        temporary.replace(output)
        if apply:
            asset = self.adapter.spec.asset_root
            target = _versioned_path(asset / "profiles/fitting_profile.json", stamp)
            backup = self.adapter.spec.generated_dir / "fitting_history" / target.stem.removeprefix("fitting_profile_")
            backup.mkdir(parents=True)
            config_path = asset / "mounting.json"
            config = json.loads(config_path.read_text())
            shutil.copy2(config_path, backup / "mounting.json")
            if config.get("fitting_profile") and (asset / config["fitting_profile"]).is_file():
                shutil.copy2(asset / config["fitting_profile"], backup / "previous_profile.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_suffix(".json.tmp")
            staged.write_text(json.dumps(profile, indent=2) + "\n")
            staged.replace(target)
            config["fitting_profile"] = target.relative_to(asset).as_posix()
            staged = config_path.with_suffix(".json.tmp")
            staged.write_text(json.dumps(config, indent=2) + "\n")
            staged.replace(config_path)
        return output

    def load(self, path: Path) -> None:
        """Load draft geometry and contact points without modifying default assets."""
        document = json.loads(Path(path).read_text())
        output = self.adapter.spec.generated_dir / "fitting/loaded_draft.xml"
        build_model(output, fitting_profile=document)
        loaded = type(self)(self.adapter, model_path=output)
        self.__dict__.update(loaded.__dict__)
