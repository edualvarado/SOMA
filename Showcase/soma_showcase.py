"""SOMA showcase in Blender: input motion -> individual muscle deformation, posed and in the canonical T-pose.

Build the scene from a cache of build_showcase_cache.py (Blender 4.2+):
    blender --python Showcase/soma_showcase.py -- --cache Showcase/cache/S1_shot_003 [--save Showcase/SOMA_S1_shot_003.blend]

The script is stored in the .blend as an auto-run module (allow "Auto Run Python Scripts" when opening it): it registers
the frame handler that streams the cache (memory-mapped .npy, nothing is computed per frame except lbs + s * d) and the
"SOMA" panel in the 3D view sidebar (N): sequence (load a cache, or generate one for any subject / shot), deformation
scale (0 = LBS, 1 = SOMA), colour by muscle or |SOMA - LBS| heatmap, fill / wire, visibility of skin / bones / skeleton.
Generating runs 03_evaluate_metrics.py --export-showcase and build_showcase_cache.py in the background with the
repository's Python environment (SKIM_ROOT and SOMA_PYTHON, else `python` on PATH, are read when the scene is built).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import threading

import bpy
import numpy as np
from mathutils import Matrix
from bpy.app.handlers import persistent

STAGES = {  # name: (x offset in the model frame (m), posed?)
    "Input": (-1.6, True),
    "Posed": (0.0, True),
    "TPose": (2.0, False),
}
LABELS = {"Input": "Input motion", "Posed": "Muscles (posed)", "TPose": "Muscles (canonical T-pose)"}
SUBJECTS = ("S1", "S2", "S3", "S4", "S5")
CHECKPOINT = "checkpoints/{S}_ABLATION_UNET_RELEASE_V2_e40_best_data.pth"
_CACHE = {}
_JOB = {"thread": None, "msg": "", "error": False, "cache": None}
_ITEMS = {}  # keeps the strings of the dynamic enum items alive (Blender requirement)


# ---------------------------------------------------------------------------------------------------------------------
# Runtime: cache streaming
# ---------------------------------------------------------------------------------------------------------------------
def _cache(scene):
    path = bpy.path.abspath(scene.get("soma_cache", ""))
    if _CACHE.get("path") != path:
        _CACHE.clear()
        if not os.path.isdir(path):
            return None
        ld = lambda n: np.load(os.path.join(path, n + ".npy"), mmap_mode="r")
        st = np.load(os.path.join(path, "static.npz"))
        _CACHE.update(path=path, static={k: st[k] for k in st.files}, A=np.load(os.path.join(path, "A.npy")),
                      joints=np.load(os.path.join(path, "joints_posed.npy")),
                      **{n: ld(n) for n in ("muscles_tpose_d", "muscles_lbs", "muscles_posed_d", "skin_tpose", "skin_posed")})
    return _CACHE


def _set_co(obj, co, disp=None):
    me = obj.data
    me.vertices.foreach_set("co", np.ascontiguousarray(co, dtype=np.float32).ravel())
    if disp is not None:
        me.attributes["disp"].data.foreach_set("value", disp.astype(np.float32))
    me.update()


def update_frame(scene, *_):
    c = _cache(scene)
    if c is None:
        return
    n = c["A"].shape[0]
    f = min(max(scene.frame_current - 1, 0), n - 1)                   # scene frame 1 = cache (video) frame 0
    s = scene.soma_scale
    rest = c["static"]["muscle_rest"]
    dT = np.asarray(c["muscles_tpose_d"][f], dtype=np.float32)
    dP = np.asarray(c["muscles_posed_d"][f], dtype=np.float32)
    objs = bpy.data.objects
    # Heatmap = |SOMA - LBS| of each character in its own space (posed / canonical)
    if "SOMA_Posed_Muscles" in objs:
        _set_co(objs["SOMA_Posed_Muscles"], c["muscles_lbs"][f] + s * dP, np.linalg.norm(dP, axis=1) * s)
    if "SOMA_TPose_Muscles" in objs:
        _set_co(objs["SOMA_TPose_Muscles"], rest + s * dT, np.linalg.norm(dT, axis=1) * s)
    for stage, (_, posed) in STAGES.items():
        sk = objs.get(f"SOMA_{stage}_Skin")
        if sk is not None and sk.visible_get():
            _set_co(sk, c["skin_posed" if posed else "skin_tpose"][f])
        if posed:
            for j in range(c["A"].shape[1]):
                b = objs.get(f"SOMA_{stage}_Bone_{j:02d}")
                if b is not None:
                    M = np.eye(4)
                    M[:3] = c["A"][f, j]
                    b.matrix_basis = Matrix(M.tolist())
            st = objs.get(f"SOMA_{stage}_Skeleton")
            if st is not None:
                _set_co(st, c["joints"][f])


def _refresh(self, context):
    for stage, mode in (("Posed", self.soma_color_posed), ("TPose", self.soma_color_tpose)):
        mat = bpy.data.materials.get(f"SOMA_Muscles_{stage}")
        if mat is not None:
            mat.node_tree.nodes["ColorMode"].outputs[0].default_value = 1.0 if mode == "DISP" else 0.0
            mat.node_tree.nodes["FillAlpha"].outputs[0].default_value = self.soma_fill
            mat.node_tree.nodes["WireStrength"].outputs[0].default_value = self.soma_wire
            mat.node_tree.nodes["DispMax"].outputs[0].default_value = self.soma_disp_max / 1000.0
        for o in [o for o in bpy.data.objects if o.name.startswith(f"SOMA_{stage}_Legend")]:
            o.hide_viewport = o.hide_render = mode != "DISP"
        t = bpy.data.objects.get(f"SOMA_{stage}_LegendMax")
        if t is not None:
            t.data.body = f"{self.soma_disp_max:g} mm"
    for o in bpy.data.objects:
        if o.name.startswith("SOMA_"):
            if "_Skin" in o.name:
                o.hide_viewport = o.hide_render = not self.soma_show_skin
            elif "_Bone_" in o.name:
                o.hide_viewport = o.hide_render = not self.soma_show_bones
            elif o.name.endswith("_Skeleton"):
                o.hide_viewport = o.hide_render = not self.soma_show_skeleton
    update_frame(context.scene)


# ---------------------------------------------------------------------------------------------------------------------
# Runtime: sequences (load a cache / generate one in the background)
# ---------------------------------------------------------------------------------------------------------------------
def _cache_root(sc):
    return os.path.join(bpy.path.abspath(sc.soma_repo), "Showcase", "cache")


def _held_out(repo, subject):
    try:
        with open(os.path.join(bpy.path.abspath(repo), "05-Training", f"{subject}_validation_filepaths.json")) as f:
            return {fid.split("_frame_")[0] for fid in json.load(f)}
    except (OSError, ValueError):
        return set()


def _title(sc, cache):
    """'S1 · shot_003 (held out)' from a cache folder named <S>_<shot>."""
    name = os.path.basename(os.path.normpath(cache)) if cache else ""
    if "_" not in name:
        return name
    subject, shot = name.split("_", 1)
    return f"{subject} · {shot}" + (" (held out)" if shot in _held_out(sc.soma_repo, subject) else "")


def _cache_items(self, context):
    sc = context.scene
    root = _cache_root(sc)
    names = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "*", "meta.json")))
    _ITEMS["cache"] = [(n, _title(sc, n), os.path.join(root, n)) for n in names] or [("NONE", "No cache yet", "")]
    return _ITEMS["cache"]


def _shot_items(self, context):
    sc = context.scene
    shots = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(
        bpy.path.abspath(sc.soma_skim_root), sc.soma_subject, "preprocessed_vFinal_clean", "shot_*.npz")))
    held, root = _held_out(sc.soma_repo, sc.soma_subject), _cache_root(sc)
    _ITEMS["shot"] = [(n, n + (" (held out)" if n in held else "") +
                       (" - cached" if os.path.isfile(os.path.join(root, f"{sc.soma_subject}_{n}", "meta.json")) else ""), "")
                      for n in shots] or [("NONE", "No shots (check the dataset path)", "")]
    return _ITEMS["shot"]


def _redraw():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type == "VIEW_3D":
                a.tag_redraw()


def _run_pipeline(repo, skim, python, subject, shot, cache):
    """Background thread: model export (03 --export-showcase) + cache build; output in <cache>.log."""
    model = cache + "_model.npz"
    env = dict(os.environ, SKIM_ROOT=skim)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    steps = [("1/2 model", [python, "03_evaluate_metrics.py", "--subject", subject, "--arch", "unet",
                            "--checkpoint", CHECKPOINT.format(S=subject), "--shot", shot, "--export-showcase", model],
              os.path.join(repo, "05-Training")),
             ("2/2 cache", [python, os.path.join("Showcase", "build_showcase_cache.py"), "--model", model,
                            "--subject", subject, "--out", cache], repo)]
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache + ".log", "w") as log:
            for label, cmd, cwd in steps:
                _JOB["msg"] = f"{subject} {shot}: {label}..."
                if subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags).returncode:
                    raise RuntimeError(f"Step {label} failed, see {os.path.basename(cache)}.log in the cache folder")
        _JOB.update(msg=f"{subject} {shot}: done", error=False, cache=cache)
    except Exception as e:
        _JOB.update(msg=str(e), error=True, cache=None)


def _poll_job():
    _redraw()
    if _JOB["thread"] is not None and _JOB["thread"].is_alive():
        return 1.0
    _JOB["thread"] = None
    cache, _JOB["cache"] = _JOB["cache"], None
    if cache:
        load_cache(bpy.context.scene, cache)
    _redraw()
    return None


class SOMA_OT_load_cache(bpy.types.Operator):
    bl_idname = "soma.load_cache"
    bl_label = "Load sequence"
    bl_description = "Load the selected cache into the scene"

    def execute(self, context):
        sc = context.scene
        if sc.soma_cache_sel in ("", "NONE"):
            self.report({"WARNING"}, "No cache to load: generate one first")
            return {"CANCELLED"}
        load_cache(sc, os.path.join(_cache_root(sc), sc.soma_cache_sel))
        return {"FINISHED"}


class SOMA_OT_generate(bpy.types.Operator):
    bl_idname = "soma.generate"
    bl_label = "Generate and load"
    bl_description = "Run the model on the selected shot and build its cache in the background, then load it"

    def execute(self, context):
        sc = context.scene
        if sc.soma_shot in ("", "NONE"):
            self.report({"WARNING"}, "No shot selected (check the dataset path)")
            return {"CANCELLED"}
        cache = os.path.join(_cache_root(sc), f"{sc.soma_subject}_{sc.soma_shot}")
        if os.path.isfile(os.path.join(cache, "meta.json")):
            load_cache(sc, cache)
            return {"FINISHED"}
        _JOB.update(msg=f"{sc.soma_subject} {sc.soma_shot}: starting...", error=False, cache=None)
        _JOB["thread"] = threading.Thread(target=_run_pipeline, daemon=True, args=(
            bpy.path.abspath(sc.soma_repo), bpy.path.abspath(sc.soma_skim_root), bpy.path.abspath(sc.soma_python),
            sc.soma_subject, sc.soma_shot, cache))
        _JOB["thread"].start()
        bpy.app.timers.register(_poll_job, first_interval=1.0)
        return {"FINISHED"}


class SOMA_PT_panel(bpy.types.Panel):
    bl_label = "SOMA Showcase"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "SOMA"

    def draw(self, context):
        sc, col = context.scene, self.layout.column()
        box = col.box()
        box.label(text=_title(sc, sc.get("soma_cache", "")) or "No sequence loaded", icon="SEQUENCE")
        row = box.row(align=True)
        row.prop(sc, "soma_cache_sel", text="")
        row.operator("soma.load_cache", text="Load")
        sub = box.box()
        sub.label(text="Generate a sequence")
        sub.prop(sc, "soma_subject")
        sub.prop(sc, "soma_shot")
        r = sub.row()
        r.enabled = _JOB["thread"] is None
        r.operator("soma.generate", icon="PLAY")
        if _JOB["msg"]:
            sub.label(text=_JOB["msg"], icon="ERROR" if _JOB["error"] else "INFO")
        sub.prop(sc, "soma_show_paths", toggle=True)
        if sc.soma_show_paths:
            sub.prop(sc, "soma_repo")
            sub.prop(sc, "soma_skim_root")
            sub.prop(sc, "soma_python")
        col.separator()
        col.prop(sc, "soma_scale", slider=True)
        row = col.row(align=True)
        row.operator("soma.set_scale", text="LBS").value = 0.0
        row.operator("soma.set_scale", text="SOMA").value = 1.0
        row.operator("soma.set_scale", text="SOMA x5").value = 5.0
        col.separator()
        col.label(text="Posed colour")
        col.row().prop(sc, "soma_color_posed", expand=True)
        col.label(text="T-pose colour")
        col.row().prop(sc, "soma_color_tpose", expand=True)
        col.prop(sc, "soma_disp_max")
        col.prop(sc, "soma_fill", slider=True)
        col.prop(sc, "soma_wire", slider=True)
        col.separator()
        col.prop(sc, "soma_show_skin")
        col.prop(sc, "soma_show_bones")
        col.prop(sc, "soma_show_skeleton")


class SOMA_OT_set_scale(bpy.types.Operator):
    bl_idname = "soma.set_scale"
    bl_label = "Set deformation scale"
    value: bpy.props.FloatProperty()

    def execute(self, context):
        context.scene.soma_scale = self.value
        return {"FINISHED"}


@persistent
def _on_frame(scene, *args):
    update_frame(scene)


@persistent
def _on_load(*args):
    update_frame(bpy.context.scene)


def register():
    S = bpy.types.Scene
    S.soma_scale = bpy.props.FloatProperty(name="Deformation scale", description="0 = LBS baseline, 1 = SOMA",
                                           default=1.0, min=0.0, soft_max=10.0, update=_refresh)
    modes = [("MUSCLE", "Muscle", "One colour per muscle"), ("DISP", "Heatmap", "|SOMA - LBS| per vertex")]
    S.soma_color_posed = bpy.props.EnumProperty(name="Posed colour", items=modes, update=_refresh)
    S.soma_color_tpose = bpy.props.EnumProperty(name="T-pose colour", items=modes, update=_refresh)
    S.soma_disp_max = bpy.props.FloatProperty(name="Heatmap max (mm)", default=10.0, min=0.1, update=_refresh)
    S.soma_fill = bpy.props.FloatProperty(name="Fill opacity", default=1.0, min=0.0, max=1.0, update=_refresh)
    S.soma_wire = bpy.props.FloatProperty(name="Wire strength", default=0.45, min=0.0, max=1.0, update=_refresh)
    S.soma_show_skin = bpy.props.BoolProperty(name="Skin", default=True, update=_refresh)
    S.soma_show_bones = bpy.props.BoolProperty(name="Bones", default=True, update=_refresh)
    S.soma_show_skeleton = bpy.props.BoolProperty(name="Skeleton (input)", default=True, update=_refresh)
    S.soma_cache_sel = bpy.props.EnumProperty(name="Sequence", items=_cache_items)
    S.soma_subject = bpy.props.EnumProperty(name="Subject", items=[(x, x, "") for x in SUBJECTS])
    S.soma_shot = bpy.props.EnumProperty(name="Shot", items=_shot_items)
    S.soma_show_paths = bpy.props.BoolProperty(name="Paths", default=False)
    S.soma_repo = bpy.props.StringProperty(name="Repository", subtype="DIR_PATH")
    S.soma_skim_root = bpy.props.StringProperty(name="Dataset", subtype="DIR_PATH", description="SKIM dataset root")
    S.soma_python = bpy.props.StringProperty(name="Python", subtype="FILE_PATH",
                                             description="Python of the SOMA environment (with torch)")
    for c in (SOMA_PT_panel, SOMA_OT_set_scale, SOMA_OT_load_cache, SOMA_OT_generate):
        if not hasattr(bpy.types, c.__name__):
            bpy.utils.register_class(c)
    for h, fn in ((bpy.app.handlers.frame_change_pre, _on_frame), (bpy.app.handlers.load_post, _on_load)):
        for f in [f for f in h if getattr(f, "__name__", "") == fn.__name__]:
            h.remove(f)
        h.append(fn)


# ---------------------------------------------------------------------------------------------------------------------
# Build: scene (once) and characters (per cache)
# ---------------------------------------------------------------------------------------------------------------------
def _mesh(name, verts, faces_flat, sizes, parent, mat, coll):
    faces = np.split(np.asarray(faces_flat), np.cumsum(sizes)[:-1])
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts, dtype=np.float64).tolist(), [], [f.tolist() for f in faces])
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    ob.parent = parent
    me.materials.append(mat)
    return ob


def _heat_ramp(cr):
    """|SOMA - LBS| colour map (magma-like, never black: it stays visible on the black background)."""
    for e, (pos, col) in zip([cr.elements[0], cr.elements[1], cr.elements.new(0.4), cr.elements.new(0.75)],
                             [(0.0, (0.10, 0.05, 0.45)), (1.0, (1.0, 0.97, 0.45)), (0.4, (0.80, 0.10, 0.45)),
                              (0.75, (1.0, 0.45, 0.05))]):
        e.position, e.color = pos, (*col, 1)


def _muscle_material(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.surface_render_method = "DITHERED"                   # depth-correct; BLENDED is unsorted within an object (patches)
    m.use_backface_culling = False
    nt, N = m.node_tree, m.node_tree.nodes
    N.clear()
    def node(t, name=None, loc=(0, 0), **kw):
        n = N.new(t)
        n.location = loc
        if name:
            n.name = n.label = name
        for k, v in kw.items():
            setattr(n, k, v)
        return n
    col = node("ShaderNodeAttribute", "MuscleColor", (-900, 200), attribute_name="muscle_color")
    dsp = node("ShaderNodeAttribute", "Disp", (-1100, -100), attribute_name="disp")
    dmax = node("ShaderNodeValue", "DispMax", (-1100, -300)); dmax.outputs[0].default_value = 0.01
    div = node("ShaderNodeMath", None, (-900, -150), operation="DIVIDE", use_clamp=True)
    ramp = node("ShaderNodeValToRGB", None, (-700, -150))
    _heat_ramp(ramp.color_ramp)
    mode = node("ShaderNodeValue", "ColorMode", (-700, 50)); mode.outputs[0].default_value = 0.0
    mix = node("ShaderNodeMix", None, (-400, 100), data_type="RGBA")
    wire = node("ShaderNodeWireframe", None, (-400, -250), use_pixel_size=True); wire.inputs[0].default_value = 1.0
    fill = node("ShaderNodeValue", "FillAlpha", (-400, -400)); fill.outputs[0].default_value = 1.0
    alpha = node("ShaderNodeMath", None, (-150, -300), operation="MAXIMUM")
    bsdf = node("ShaderNodeBsdfPrincipled", None, (-100, 200))
    bsdf.inputs["Roughness"].default_value = 0.45
    bsdf.inputs["Emission Strength"].default_value = 0.25              # keeps the colours saturated in shadow
    white = node("ShaderNodeEmission", None, (-100, -50)); white.inputs["Strength"].default_value = 1.0
    wstr = node("ShaderNodeValue", "WireStrength", (-400, -100)); wstr.outputs[0].default_value = 0.45
    wmul = node("ShaderNodeMath", None, (-150, -150), operation="MULTIPLY")
    white.inputs["Color"].default_value = (1, 1, 1, 1)
    sh_mix = node("ShaderNodeMixShader", None, (200, 100))
    transp = node("ShaderNodeBsdfTransparent", None, (200, -150))
    a_mix = node("ShaderNodeMixShader", None, (450, 0))
    out = node("ShaderNodeOutputMaterial", None, (650, 0))
    L = nt.links.new
    L(dsp.outputs["Fac"], div.inputs[0]); L(dmax.outputs[0], div.inputs[1]); L(div.outputs[0], ramp.inputs[0])
    L(mode.outputs[0], mix.inputs["Factor"]); L(col.outputs["Color"], mix.inputs["A"]); L(ramp.outputs["Color"], mix.inputs["B"])
    L(mix.outputs["Result"], bsdf.inputs["Base Color"]); L(mix.outputs["Result"], bsdf.inputs["Emission Color"])
    L(wire.outputs[0], wmul.inputs[0]); L(wstr.outputs[0], wmul.inputs[1])
    L(wmul.outputs[0], sh_mix.inputs[0]); L(bsdf.outputs[0], sh_mix.inputs[1]); L(white.outputs[0], sh_mix.inputs[2])
    L(wire.outputs[0], alpha.inputs[0]); L(fill.outputs[0], alpha.inputs[1])
    L(alpha.outputs[0], a_mix.inputs[0]); L(transp.outputs[0], a_mix.inputs[1]); L(sh_mix.outputs[0], a_mix.inputs[2])
    L(a_mix.outputs[0], out.inputs["Surface"])
    return m


def _legend(stage, dx, parent_coll, text_mat):
    """Colour bar under a character, shown while its heatmap is on."""
    bar = bpy.data.meshes.new(f"SOMA_{stage}_Legend")
    bar.from_pydata([(-0.4, 0, 0), (0.4, 0, 0), (0.4, 0, 0.035), (-0.4, 0, 0.035)], [], [(0, 1, 2, 3)])
    uv = bar.uv_layers.new(); uv.data.foreach_set("uv", [0, 0, 1, 0, 1, 1, 0, 1])
    m = bpy.data.materials.new(f"SOMA_{stage}_Legend")
    m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links.new
    N.clear()
    tc, sep, ramp = N.new("ShaderNodeUVMap"), N.new("ShaderNodeSeparateXYZ"), N.new("ShaderNodeValToRGB")
    em, out = N.new("ShaderNodeEmission"), N.new("ShaderNodeOutputMaterial")
    _heat_ramp(ramp.color_ramp)
    L(tc.outputs[0], sep.inputs[0]); L(sep.outputs[0], ramp.inputs[0]); L(ramp.outputs[0], em.inputs[0]); L(em.outputs[0], out.inputs[0])
    bar.materials.append(m)
    ob = bpy.data.objects.new(f"SOMA_{stage}_Legend", bar)
    parent_coll.objects.link(ob)
    ob.location = (dx, 0, -0.2)
    for suffix, body, x, align in (("Min", "0", -0.4, "RIGHT"), ("Max", "10 mm", 0.4, "LEFT"), ("Title", "|SOMA - LBS|", 0.0, "CENTER")):
        t = bpy.data.curves.new(f"SOMA_{stage}_Legend{suffix}", "FONT")
        t.body, t.align_x, t.size = body, align, 0.06
        to = bpy.data.objects.new(f"SOMA_{stage}_Legend{suffix}", t)
        parent_coll.objects.link(to)
        to.location = (dx + x + (-0.03 if align == "RIGHT" else 0.03 if align == "LEFT" else 0), 0, -0.195 if suffix != "Title" else -0.14)
        to.rotation_euler = (np.pi / 2, 0, 0)
        t.materials.append(text_mat)


def _simple_material(name, color, alpha=1.0, emission=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Alpha"].default_value = alpha
    b.inputs["Roughness"].default_value = 0.6
    if emission:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emission
    if alpha < 1:
        m.surface_render_method = "BLENDED"
    m.diffuse_color = (*color, alpha)
    return m


def _stick_nodes(mat):
    ng = bpy.data.node_groups.new("SOMA_Stick", "GeometryNodeTree")
    ng.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    N, L = ng.nodes, ng.links.new
    gi, go = N.new("NodeGroupInput"), N.new("NodeGroupOutput")
    m2c = N.new("GeometryNodeMeshToCurve")
    circ = N.new("GeometryNodeCurvePrimitiveCircle"); circ.inputs["Radius"].default_value = 0.012; circ.inputs["Resolution"].default_value = 10
    c2m = N.new("GeometryNodeCurveToMesh")
    sph = N.new("GeometryNodeMeshUVSphere"); sph.inputs["Radius"].default_value = 0.024
    sph.inputs["Segments"].default_value = 16; sph.inputs["Rings"].default_value = 8
    inst = N.new("GeometryNodeInstanceOnPoints")
    join = N.new("GeometryNodeJoinGeometry")
    setm = N.new("GeometryNodeSetMaterial"); setm.inputs["Material"].default_value = mat
    L(gi.outputs[0], m2c.inputs["Mesh"]); L(m2c.outputs[0], c2m.inputs["Curve"]); L(circ.outputs["Curve"], c2m.inputs["Profile Curve"])
    L(gi.outputs[0], inst.inputs["Points"]); L(sph.outputs["Mesh"], inst.inputs["Instance"])
    L(c2m.outputs[0], join.inputs[0]); L(inst.outputs[0], join.inputs[0])
    L(join.outputs[0], setm.inputs["Geometry"]); L(setm.outputs[0], go.inputs[0])
    return ng


def _content():
    """Collection holding the characters of the loaded sequence; emptied on every load."""
    coll = bpy.data.collections.get("SOMA_Content")
    if coll is None:
        coll = bpy.data.collections.new("SOMA_Content")
        bpy.context.scene.collection.children.link(coll)
    return coll


def load_cache(sc, cache):
    """Replace the characters with those of a cache (subjects differ in meshes) and set the frame range."""
    cache = os.path.abspath(cache)
    meta = json.load(open(os.path.join(cache, "meta.json")))
    coll = _content()
    for ob in list(coll.objects):
        data = ob.data
        bpy.data.objects.remove(ob)
        if isinstance(data, bpy.types.Mesh) and data.users == 0:
            bpy.data.meshes.remove(data)
    sc["soma_cache"] = cache
    S = _cache(sc)["static"]
    sc.frame_start, sc.frame_end = 1, meta["frames"]
    sc.render.fps = int(meta["fps"])
    mats = bpy.data.materials

    # Per-vertex muscle colour + per-frame displacement attribute
    vcol = np.zeros((len(S["muscle_rest"]), 4), np.float32); vcol[:, 3] = 1
    for (a, b), col in zip(S["muscle_ranges"], S["muscle_colors"]):
        vcol[a:b, :3] = col
    bj = S["bone_joint"]
    bf = np.split(S["bone_faces"], np.cumsum(S["bone_face_sizes"])[:-1])
    for stage, (dx, posed) in STAGES.items():
        root = bpy.data.objects.new(f"SOMA_{stage}", None)
        coll.objects.link(root)
        root.rotation_euler = (np.pi / 2, 0, 0)                      # model frame is Y-up
        root.location = (dx, 0, 0)
        if stage != "Input":
            mus = _mesh(f"SOMA_{stage}_Muscles", S["muscle_rest"], S["muscle_faces"], S["muscle_face_sizes"], root,
                        mats[f"SOMA_Muscles_{stage}"], coll)
            a = mus.data.color_attributes.new("muscle_color", "FLOAT_COLOR", "POINT")
            a.data.foreach_set("color", vcol.ravel())
            mus.data.attributes.new("disp", "FLOAT", "POINT")
            mus.data.shade_smooth()
        else:
            _mesh(f"SOMA_{stage}_Skin", S["skin_rest"], S["skin_faces"], S["skin_face_sizes"], root, mats["SOMA_Skin"],
                  coll).data.shade_smooth()
            edges = [(j, int(p)) for j, p in enumerate(S["parents"]) if p >= 0]
            me = bpy.data.meshes.new(f"SOMA_{stage}_Skeleton")
            me.from_pydata(S["j_rest"].tolist(), edges, [])
            sk = bpy.data.objects.new(f"SOMA_{stage}_Skeleton", me)
            coll.objects.link(sk)
            sk.parent = root
            sk.modifiers.new("Stick", "NODES").node_group = bpy.data.node_groups["SOMA_Stick"]
        for j in sorted(set(bj.tolist())):
            keep = [f for f in bf if bj[f[0]] == j]
            used = np.unique(np.concatenate(keep)); remap = -np.ones(len(bj), int); remap[used] = np.arange(len(used))
            ff = np.concatenate([remap[f] for f in keep])
            _mesh(f"SOMA_{stage}_Bone_{j:02d}", S["bone_rest"][used], ff, [len(f) for f in keep], root,
                  mats["SOMA_Bone"], coll).data.shade_smooth()
    title = bpy.data.objects.get("SOMA_Title")
    if title is not None:
        title.data.body = _title(sc, cache)
    if os.path.dirname(cache) == os.path.abspath(_cache_root(sc)):
        sc.soma_cache_sel = os.path.basename(cache)
    sc.frame_set(1)
    _refresh(sc, bpy.context)
    print(f"[SOMA] loaded {cache}")


def setup_scene(sc):
    """Render settings, materials, labels, camera and lights (once); the characters come from load_cache()."""
    sc.sync_mode = "FRAME_DROP"
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
    sc.view_settings.view_transform = "Standard"                      # AgX desaturates the muscle colours
    sc.soma_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sc.soma_skim_root = os.environ.get("SKIM_ROOT", "")
    sc.soma_python = os.environ.get("SOMA_PYTHON") or shutil.which("python") or "python"

    world = bpy.data.worlds.new("SOMA_World"); sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)

    m_text = _simple_material("SOMA_Text", (0.92, 0.92, 0.92), emission=1.0)
    _simple_material("SOMA_Skin", (0.85, 0.75, 0.68), alpha=0.08)
    _simple_material("SOMA_Bone", (0.75, 0.72, 0.65))
    _stick_nodes(_simple_material("SOMA_Stick", (0.2, 0.75, 1.0), emission=2.0))
    for stage, (dx, _) in STAGES.items():
        if stage != "Input":
            _muscle_material(f"SOMA_Muscles_{stage}")
            _legend(stage, dx, sc.collection, m_text)
        t = bpy.data.curves.new(f"SOMA_{stage}_Label", "FONT")
        t.body, t.align_x, t.size = LABELS[stage], "CENTER", 0.09
        to = bpy.data.objects.new(f"SOMA_{stage}_Label", t)
        sc.collection.objects.link(to)
        to.location, to.rotation_euler = (dx, 0, 2.0), (np.pi / 2, 0, 0)
        to.data.materials.append(m_text)
    t = bpy.data.curves.new("SOMA_Title", "FONT")
    t.align_x, t.size = "CENTER", 0.12
    to = bpy.data.objects.new("SOMA_Title", t)
    sc.collection.objects.link(to)
    to.location, to.rotation_euler = (0.25, 0, 2.3), (np.pi / 2, 0, 0)
    t.materials.append(m_text)

    cam = bpy.data.objects.new("SOMA_Camera", bpy.data.cameras.new("SOMA_Camera"))
    sc.collection.objects.link(cam); sc.camera = cam
    cam.location, cam.rotation_euler = (0.25, -6.2, 1.05), (np.pi / 2, 0, 0)
    cam.data.lens = 40
    for name, loc, rot, energy in [("Key", (-3, -5, 5), (0.9, 0, -0.5), 500), ("Fill", (4, -4, 3), (1.1, 0, 0.8), 200),
                                   ("Rim", (0, 5, 4), (-0.9, 0, 0), 300)]:
        l = bpy.data.objects.new(f"SOMA_{name}", bpy.data.lights.new(f"SOMA_{name}", "AREA"))
        l.data.energy, l.data.size = energy, 4
        l.location, l.rotation_euler = loc, rot
        sc.collection.objects.link(l)


def build(cache, save):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    register()
    sc = bpy.context.scene
    setup_scene(sc)
    load_cache(sc, cache)
    # Store this script in the .blend as an auto-run module (re-registers the player on load)
    txt = bpy.data.texts.load(os.path.abspath(__file__))
    txt.use_module = True
    if save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(save))
        print(f"[SOMA] saved {save}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--cache" in argv:
        cache = argv[argv.index("--cache") + 1]
        save = argv[argv.index("--save") + 1] if "--save" in argv else None
        build(cache, save)
    else:
        register()
else:
    register()
