# Showcase

Real-time Blender viewer of SOMA's individual muscle deformation: the input motion, the 95 individual muscle
meshes posed, and the same muscles in the canonical T-pose, each muscle in its own colour (wireframe) or as a heatmap
of |SOMA − LBS|.

Nothing is computed while playing: the model output of a shot is precomputed once into a local cache that Blender
streams frame by frame. The cache is not distributed; build it from the dataset and the pretrained models in three
steps (about a minute in total on a GPU).

---

## Files

| File | Purpose |
|------|---------|
| `build_showcase_cache.py` | Binds the individual muscles to the muscle layer (`05-Training/muscle_binding.py`), poses them with the model's per-joint transforms and writes the cache (memory-mapped `.npy`) |
| `soma_showcase.py` | Builds the Blender scene from a cache and saves a `.blend`; also the player (frame handler + panel), stored in the `.blend` |

---

## Usage

Requirements: the SKIM dataset, the pretrained models (`SOMA_checkpoints.tar` in `05-Training/checkpoints/`), the
Python environment of the repository and [Blender](https://www.blender.org/) 4.2 or later.

```bash
export SKIM_ROOT=/path/to/SKIM_dataset

# 1. Model output of one shot (any subject / shot; S1 shot_003 is held out from training)
cd 05-Training
python 03_evaluate_metrics.py --subject S1 --arch unet \
    --checkpoint checkpoints/S1_ABLATION_UNET_RELEASE_V2_e40_best_data.pth \
    --shot shot_003 --export-showcase ../Showcase/cache/S1_shot_003_model.npz
cd ..

# 2. Cache: individual muscles (T-pose + posed), skin, skeleton
python Showcase/build_showcase_cache.py --model Showcase/cache/S1_shot_003_model.npz \
    --subject S1 --out Showcase/cache/S1_shot_003

# 3. Blender scene (from the same shell, with the SOMA environment active)
blender --python Showcase/soma_showcase.py -- --cache Showcase/cache/S1_shot_003 \
    --save Showcase/SOMA_showcase.blend
```

The cache takes about 1.7 MB per frame (S1 shot_003, 1937 frames: 3.2 GB) and is ignored by git (`Showcase/cache/`).

Open the `.blend` and allow **Auto Run Python Scripts** when Blender asks: the player is a script stored in the file
and re-registers itself on load. Press Space to play (25 fps).

**Other subjects and shots** can then be generated from Blender itself (panel → *Generate a sequence*): pick the
subject and the shot (held-out shots are marked) and press *Generate and load*. Steps 1–2 run in the background
(about 30 s on a GPU) and the sequence is loaded when ready; the caches already built are listed in the *Sequence*
menu. Generating needs the repository, the dataset and the Python of the SOMA environment: their paths are taken from
`SKIM_ROOT` and `SOMA_PYTHON` (else `python` on `PATH`) when step 3 runs, and can be changed under *Paths*.

---

## Controls

3D view sidebar (N) → **SOMA**:

| Control | Effect |
|---------|--------|
| Sequence + Load | Switch to any cache already built |
| Generate a sequence | Subject + shot → build its cache in the background and load it; *Paths*: repository, dataset, Python |
| Deformation scale | 0 = LBS baseline, 1 = SOMA, > 1 exaggerated. Exact: LBS is linear in the vertex, so the posed muscles are `LBS + s · (SOMA − LBS)` |
| LBS / SOMA / SOMA x5 | Shortcuts for the scale |
| Posed colour, T-pose colour | Per character: one colour per muscle, or a heatmap of \|SOMA − LBS\| (in that character's space) with a colour bar |
| Heatmap max (mm) | Top of the heatmap scale |
| Fill opacity, Wire strength | Muscle surface opacity and white wireframe intensity |
| Skin, Bones, Skeleton (input) | Visibility |

The individual muscles follow the muscle layer through `05-Training/muscle_binding.py` (Gaussian blend of the nearest
muscle-layer vertices). The posed characters stand in place: the root translation of the capture is not applied.
