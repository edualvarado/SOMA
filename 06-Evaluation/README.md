# 06-Evaluation

Sixth stage of the SOMA pipeline. Evaluates the **HIT** baseline (SMPL body with predicted inner
tissue layers) on the SOMA captures, for comparison with SOMA. SOMA's own metrics come from
`05-Training/03_evaluate_metrics.py`.

---

## Files

| File | Purpose |
|------|---------|
| `hit_bio_evaluation.py` | Biomechanical metrics of the HIT layers per frame: muscle-outside-skin intersection ratio, fat and muscle volume coefficient of variation |
| `hit_bio_evaluation_job.sh` | SLURM wrapper for `hit_bio_evaluation.py` (results are written to the submit directory) |
| `smpl_alignment.py` | Aligns the SOMA T-pose markers to the HIT/SMPL T-pose mesh and writes `soma_to_smpl_barycentric_map.json` |
| `smpl_metrics.py` | Marker error of HIT against the observed markers (MPME, MedPME, P90, DRE); paper-era inputs, see the note below |
| `soma_to_smpl_barycentric_map.json` | SOMA marker → SMPL surface barycentric map (output of `smpl_alignment.py`) |
| `hit_bio_evaluation_results_S{N}_shot_001.json` | `hit_bio_evaluation.py` results on shot_001 for each subject (summary + per frame) |

---

## Inputs

The scripts are configured by constants at the top of each file (`SUBJECT`, `SEQUENCE`, alignment
offsets) and two environment variables:

| Variable | Default | Used by | Content |
|----------|---------|---------|---------|
| `HIT_OUTPUTS` | `HIT/outputs_soma_eval` | `hit_bio_evaluation.py` | HIT results per frame: `<S>/<shot>/frame_XXXXXX/{hit_male_best,hit_female_best}/{smpl_mesh,LT_mesh,AT_mesh}.obj` |
| `SOMA_STATIC` | `static00` | `smpl_alignment.py`, `smpl_metrics.py` | Capture working tree: HIT T-pose mesh, HIT outputs, `<S>/canonical_model/`, per-frame residuals |

---

## Usage

```bash
cd 06-Evaluation
export HIT_OUTPUTS=/path/to/HIT/outputs_soma_eval
python hit_bio_evaluation.py       # -> hit_bio_evaluation_results_<S>_<shot>.json + plot
sbatch hit_bio_evaluation_job.sh   # same, on SLURM

export SOMA_STATIC=/path/to/static00
python smpl_alignment.py           # -> soma_to_smpl_barycentric_map.json
python smpl_metrics.py             # prints the HIT marker metrics
```

> **Note:** `smpl_metrics.py` is kept as used for the paper. It expects the original per-frame
> residual layout (`<S>/preprocessed_vFinal_clean/residuals/shot_001_frame_XXXX.npy`, residuals scaled
> by 0.1, T-pose space) and HIT outputs for S1 shot_001. It does not read the released SKIM dataset, and
> with the current held-out split (whole shots, no shot_001) it selects no frames.
