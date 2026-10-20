"""Hyperparameter sweep: train (01) -> evaluate (03) -> one row per run in sweep_results.csv.

Usage (from 05-Training/, with SKIM_ROOT set):
    python sweep.py --only B_mlp_data_only,C_linear_data_only
    python sweep.py --list

Runs already in the CSV are skipped (resume-safe). Only the best-by-val-data checkpoint is kept;
the per-epoch best-total checkpoints of each run are deleted after evaluation to save disk.
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import time

SUBJECT = "S1"
EPOCHS = 8
CSV_PATH = "sweep_results.csv"
LOG_DIR = "sweep_logs"

# name -> (arch, extra args for 01). The name becomes the run tag.
RUNS = {
    # Phase 1: sanity -- can the model beat LBS on markers at all?
    # All runs use --marker-offset (zero displacement == LBS); 03 is called with it too.
    "B_mlp_data_only":        ("mlp",    ["--marker-offset", "--preset", "data_only", "--data-norm", "visible"]),
    "C_linear_data_only":     ("linear", ["--marker-offset", "--preset", "data_only", "--data-norm", "visible"]),
    # Phase 2 anchor: paper final config
    "P_mlp_paper":            ("mlp",    ["--marker-offset", "--preset", "paper_final"]),
    "P_mlp_paper_vis":        ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible"]),
    # Phase 2: one factor at a time around the anchor P_mlp_paper_vis
    "D_wdata500":             ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=500"]),
    "D_wdata5000":            ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=5000"]),
    "D_wdata5e4":             ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000"]),
    "D_wdata5e5":             ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=500000"]),
    "D_lin_wdata5000":        ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=5000"]),
    "D_lin_wdata5e4":         ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000"]),
    # Phase 2b: one factor at a time around w_data=5e4 (anchor D_wdata5e4 / D_lin_wdata5e4)
    "V5e4_vol100":            ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=100", "--w", "w_vol_skin=100"]),
    "V5e4_vol1000":           ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000", "--w", "w_vol_skin=1000"]),
    "V5e4_vol1e4":            ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=10000", "--w", "w_vol_skin=10000"]),
    "V5e4_lin_vol1000":       ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000", "--w", "w_vol_skin=1000"]),
    "V5e4_vol1e5":            ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=100000", "--w", "w_vol_skin=100000"]),
    "V5e4_vol1e6":            ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000"]),
    "V5e4_lin_vol1e5":        ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=100000", "--w", "w_vol_skin=100000"]),
    "L5e4_lr3e-4":            ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--lr", "3e-4"]),
    "L5e4_lr1e-3":            ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--lr", "1e-3"]),
    "A5e4_unet":              ("unet",   ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000"]),
    # Phase 3: combine winners (w_data 5e4, vol 1e6, lr 1e-4)
    "F_lin_vol1e6":           ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000"]),
    "F_unet_vol1e6":          ("unet",   ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000"]),
    "F_mlp_vol1e6_lr5e-5":    ("mlp",    ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--lr", "5e-5"]),
    "F_lin_vol1e6_wd1e5":     ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=100000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000"]),
    "F_lin_vol1e7":           ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=10000000", "--w", "w_vol_skin=10000000"]),
    # Phase 4: per-regularizer axes around F_lin_vol1e6 (linear, w_data 5e4, vol 1e6); musc/skin ratios kept
    "R_smooth_x0.1":          ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_smooth_musc=10", "--w", "w_smooth_skin=20"]),
    "R_smooth_x10":           ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_smooth_musc=1000", "--w", "w_smooth_skin=2000"]),
    "R_bi_x0.1":              ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_biharmonic_musc=10", "--w", "w_biharmonic_skin=20"]),
    "R_bi_x10":               ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_biharmonic_musc=1000", "--w", "w_biharmonic_skin=2000"]),
    "R_spring_x100":          ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_spring_musc=100000", "--w", "w_spring_skin=100000"]),
    "R_spring_x1e4":          ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_spring_musc=10000000", "--w", "w_spring_skin=10000000"]),
    "R_tan_x10":              ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_tangent_musc=10", "--w", "w_tangent_skin=1"]),
    "R_tan_x100":             ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_tangent_musc=100", "--w", "w_tangent_skin=10"]),
    # Phase 5: new anchor smooth x0.1 (10/20); spend recovered fit on physics levers
    "G_smooth_x0.01":         ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_smooth_musc=1", "--w", "w_smooth_skin=2"]),
    "G_sm_bi_x0.1":           ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_smooth_musc=10", "--w", "w_smooth_skin=20", "--w", "w_biharmonic_musc=10", "--w", "w_biharmonic_skin=20"]),
    "G_sm_spring_x10":        ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=1000000", "--w", "w_vol_skin=1000000", "--w", "w_smooth_musc=10", "--w", "w_smooth_skin=20", "--w", "w_spring_musc=10000", "--w", "w_spring_skin=10000"]),
    "G_sm_vol1e7":            ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=10000000", "--w", "w_vol_skin=10000000", "--w", "w_smooth_musc=10", "--w", "w_smooth_skin=20"]),
    "G_sm_spring_vol1e7":     ("linear", ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=10000000", "--w", "w_vol_skin=10000000", "--w", "w_smooth_musc=10", "--w", "w_smooth_skin=20", "--w", "w_spring_musc=10000", "--w", "w_spring_skin=10000"]),
    "G_sm_spring_vol1e7_unet":("unet",   ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000", "--w", "w_vol_musc=10000000", "--w", "w_vol_skin=10000000", "--w", "w_smooth_musc=10", "--w", "w_smooth_skin=20", "--w", "w_spring_musc=10000", "--w", "w_spring_skin=10000"]),
    # Release: config A (UNet, full physics) -- run with --subjects S1,...,S5 --epochs 15
    "FINAL_A_unet":           ("unet",   ["--marker-offset", "--preset", "paper_final", "--data-norm", "visible", "--w", "w_data=50000",
                                          "--w", "w_vol_musc=10000000", "--w", "w_vol_skin=10000000", "--w", "w_smooth_musc=10", "--w", "w_smooth_skin=20",
                                          "--w", "w_spring_musc=10000", "--w", "w_spring_skin=10000"]),
}

# Phase 6: physics vs LBS on S1 (UNet, FINAL_A weights). Rigid bones = what 03 evaluates; 'rel' volume
# loss is ~ (%change)^2, so its weights live around 1-1000 instead of 1e7.
_A = RUNS["FINAL_A_unet"][1]
_REL = ["--rigid-bone", "--vol-loss", "rel"]
RUNS.update({
    "P_rigid":              ("unet", _A + ["--rigid-bone"]),
    "P_rel_v10":            ("unet", _A + _REL + ["--w", "w_vol_musc=10", "--w", "w_vol_skin=10"]),
    "P_rel_v100":           ("unet", _A + _REL + ["--w", "w_vol_musc=100", "--w", "w_vol_skin=100"]),
    "P_rel_v1000":          ("unet", _A + _REL + ["--w", "w_vol_musc=1000", "--w", "w_vol_skin=1000"]),
    "P_rel_v100_fatposed":  ("unet", _A + _REL + ["--fat-posed", "--w", "w_vol_musc=100", "--w", "w_vol_skin=100"]),
    "P_rel_v1000_fatposed": ("unet", _A + _REL + ["--fat-posed", "--w", "w_vol_musc=1000", "--w", "w_vol_skin=1000"]),
    "P_rel_v10_fatposed":   ("unet", _A + _REL + ["--fat-posed", "--w", "w_vol_musc=10", "--w", "w_vol_skin=10"]),
    "P_rel_v30_fatposed":   ("unet", _A + _REL + ["--fat-posed", "--w", "w_vol_musc=30", "--w", "w_vol_skin=30"]),
})

# Phase 7: smoothness on top of P_rel_v30_fatposed (its skin/muscle were visibly wrinkled: 12 / 9 deg roughness
# vs 1.4 for LBS). The paper checkpoint was smooth because smooth (100/200) >> w_data (50). Later --w wins.
_V30 = RUNS["P_rel_v30_fatposed"][1]
RUNS.update({
    "S_sm1e3":       ("unet", _V30 + ["--w", "w_smooth_musc=1000", "--w", "w_smooth_skin=2000"]),
    "S_sm1e4":       ("unet", _V30 + ["--w", "w_smooth_musc=10000", "--w", "w_smooth_skin=20000"]),
    "S_sm1e5":       ("unet", _V30 + ["--w", "w_smooth_musc=100000", "--w", "w_smooth_skin=200000"]),
    "S_bi1e4":       ("unet", _V30 + ["--w", "w_biharmonic_musc=10000", "--w", "w_biharmonic_skin=20000"]),
    "S_data5e3":     ("unet", _V30 + ["--w", "w_data=5000"]),
})

# Phase 8: 01's smoothness/biharmonic terms used trimesh's averaging operator A instead of the Laplacian I - A,
# which penalised smooth deformation and rewarded noise. Fixed in 01 (all runs above used the old operator:
# add --legacy-laplacian to reproduce them). Re-tune the smoothing weights on P_rel_v30_fatposed.
RUNS.update({
    "L_v30":          ("unet", _V30),
    "L_v30_sm100":    ("unet", _V30 + ["--w", "w_smooth_musc=100", "--w", "w_smooth_skin=200"]),
    "L_v30_sm1e3":    ("unet", _V30 + ["--w", "w_smooth_musc=1000", "--w", "w_smooth_skin=2000"]),
    "L_v30_bi1e3":    ("unet", _V30 + ["--w", "w_smooth_musc=100", "--w", "w_smooth_skin=200",
                                        "--w", "w_biharmonic_musc=1000", "--w", "w_biharmonic_skin=2000"]),
})

# Phase 9: the abdomen is never observed when bent (hip flex > 60 deg); what drives its shape there?
_SM100 = RUNS["L_v30_sm100"][1]
RUNS.update({
    "U_v0_sm100":     ("unet", _SM100 + ["--w", "w_vol_musc=0", "--w", "w_vol_skin=0"]),
    # Volume per patch (fat: geodesic patches of radius R; muscle: individual-muscle footprints) + inverted-prism penalty
    "B_f6_mp":        ("unet", _SM100 + ["--fat-patch-radius", "0.06", "--musc-patches",
                                         "--w", "w_inv_skin=30", "--w", "w_inv_musc=30"]),
    "B_f6":           ("unet", _SM100 + ["--fat-patch-radius", "0.06", "--w", "w_inv_skin=30"]),
    "B_mp":           ("unet", _SM100 + ["--musc-patches", "--w", "w_inv_musc=30"]),
    # Tangent loss off: let tissue slide into a fold instead of only inflating along the rest normals
    "T_tan_s0":       ("unet", _SM100 + ["--w", "w_tangent_skin=0"]),
    "T_tan_0":        ("unet", _SM100 + ["--w", "w_tangent_skin=0", "--w", "w_tangent_musc=0"]),
    # Posed crease loss: squared dihedral change vs the posed LBS mesh (skin + muscle)
    "C_cr100":        ("unet", _SM100 + ["--w", "w_crease_skin=100", "--w", "w_crease_musc=100"]),
    "C_cr150":        ("unet", _SM100 + ["--w", "w_crease_skin=150", "--w", "w_crease_musc=150"]),
    "C_cr200":        ("unet", _SM100 + ["--w", "w_crease_skin=200", "--w", "w_crease_musc=200"]),
    "C_cr300":        ("unet", _SM100 + ["--w", "w_crease_skin=300", "--w", "w_crease_musc=300"]),
    "C_cr1e3":        ("unet", _SM100 + ["--w", "w_crease_skin=1000", "--w", "w_crease_musc=1000"]),
    # Correctives in the per-vertex posed frame (blended rotation) instead of through the LBS matrix
    "P2_cr100":       ("unet", _SM100 + ["--posed-frame", "--w", "w_crease_skin=100", "--w", "w_crease_musc=100"]),
    # One-sided crease: only creasing beyond LBS is penalised (undoing LBS collapse, e.g. at the shoulder, is free)
    "C1_cr50":        ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=50", "--w", "w_crease_musc=50"]),
    "C1_cr75":        ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75"]),
    "C1_cr100":       ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=100", "--w", "w_crease_musc=100"]),
    "C1_cr300":       ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=300", "--w", "w_crease_musc=300"]),
    # Residual scale 1.0 (real-size targets) on the chosen C1_cr75; 03's MPME is in x0.1 units, judge with real-scale error
    "R1_cr75":        ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--residual-scale", "1.0"]),
    "R1s_cr75":       ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--residual-scale", "1.0", "--residual-static-removal"]),
    # Posed-space data loss (residual rotated into the model frame, compared with the posed marker shift)
    "Q1_cr75":        ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--posed-data"]),
    "Q1s_cr75":       ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--posed-data", "--residual-scale", "1.0", "--residual-static-removal"]),
    # + per-joint rigid skeleton correction inside LBS (prior w_rigid, default 1)
    "G1s_cr75":       ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--posed-data", "--residual-scale", "1.0", "--residual-static-removal", "--rigid-correction"]),
    # 2026-10-02 (late): corrected world->model rotation Ry(90)Rx(-90) (Q1/Q1s/G1s used Rx(-90) only) + rebuilt rig
    "Q2_cr75":        ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--posed-data"]),
    "N2_cr75":        ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--posed-data", "--rig", "rigs/S1_rig.npz"]),
    "N3_cr75":        ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--posed-data", "--rig", "rigs/S1_rig.npz"]),  # N2 on rig v3 (N2 = v1, discard)
    "N1s_cr75":       ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--posed-data", "--rig", "rigs/S1_rig.npz", "--residual-scale", "1.0", "--residual-static-removal"]),
    "R1_cr75_wd500":  ("unet", _SM100 + ["--crease-mode", "onesided", "--w", "w_crease_skin=75", "--w", "w_crease_musc=75", "--residual-scale", "1.0", "--w", "w_data=500"]),
})

# 2026-10-04: N1s artifacts. 60-90% of its per-segment displacement is rigid (forearm/thigh twist 15-36 deg about the
# bone axis, 20-35 mm shifts): pose error absorbed as deformation, sheared against the pinned hands.
_N1S = RUNS["N1s_cr75"][1]
RUNS.update({
    "K1_rigid":       ("unet", _N1S + ["--rigid-correction"]),
    "K2_sm_bi_x3":    ("unet", _N1S + ["--w", "w_smooth_musc=300", "--w", "w_smooth_skin=600",
                                       "--w", "w_biharmonic_musc=300", "--w", "w_biharmonic_skin=600"]),
})

# 2026-10-07: K1_rigid is the 01/03 default (preset soma_final, rig = rigs/<S>_rig.npz): any subject, no flags
RUNS["K1"] = ("unet", ["--no-drop-isolated", "--no-unobs-normal", "--w", "w_unobs=0"])  # defaults before 2026-10-11 (no unobs prior)

# 2026-10-10: bumps where markers are unobserved (d is larger in a marker's unobserved frames than in its observed ones).
# K1 + pull to LBS on vertices whose nearest markers are unobserved in the frame + drop isolated observations
RUNS.update({
    "U_unobs3e3": ("unet", ["--drop-isolated", "--no-unobs-normal", "--w", "w_unobs=3000"]),
    "U_unobs1e4": ("unet", ["--drop-isolated", "--no-unobs-normal", "--w", "w_unobs=10000"]),
    "U_unobs3e4": ("unet", ["--drop-isolated", "--no-unobs-normal", "--w", "w_unobs=30000"]),
})
# 2026-10-11: U_unobs1e4 pins hidden skin tangentially too -> sliding buttock skin bunches at the gluteal fold.
# Skin term on the normal component only (muscle unchanged)
RUNS["U_unobsN1e4"] = ("unet", ["--drop-isolated", "--unobs-normal", "--w", "w_unobs=10000"])
# 2026-10-11: final model = 01 defaults (K1 + --drop-isolated + --unobs-normal + w_unobs 1e4)
RUNS["FINAL"] = ("unet", [])
# 2026-10-11: run-to-run variance (01 is unseeded): identical configs, new tags
RUNS["K1_r2"] = RUNS["K1"]
RUNS["FINAL_r2"] = RUNS["FINAL"]
# 2026-10-12: S3 forearm bumps = the raw Captury ForeArm roll that the observed forearm does not follow.
# FINAL + ForeArm twist removed (01/03 default for S2-S5 from now on; explicit here). DT_detwist_e20 (twist moved onto
# the Hand, superseded: candy-wrapper wrists, the hand markers do not follow the twist either) ran with the older code.
RUNS["DT_drop"] = ("unet", ["--forearm-detwist"])
# 2026-10-12: S5 skin peaks at marker groups that disagree with their neighbours: drop those observations (>5 cm)
RUNS["DI_5cm"] = ("unet", ["--drop-inconsistent", "0.05"])
# 2026-10-12: RELEASE model (ECCV 2026 camera-ready) = 01 defaults, seeded. Run: --subjects S1,S2,S3,S4,S5 --epochs 40.
# K1 + --drop-isolated + --unobs-normal (w_unobs 1e4) + --drop-inconsistent 0.05 + --forearm-detwist (auto: S2-S5)
# + rebuilt rigs (rigs/<S>_rig.npz; S4 rebuilt after the layer fix) + LAYER_REST_OFFSET (S4) + global_R in every packed shot
RUNS["RELEASE"] = ("unet", ["--seed", "0"])
# 2026-10-13: RELEASE retrained on SKIM V2 (build_skim_v2.py: rig, de-twist, S4 layers baked in; fixes the S2-S5 forearm
# target bias of the V1 conversion, S3 shot_005/008 poses). Same config; SKIM_ROOT = the V2 dataset, --subjects S1,S2,S3,S4,S5 --epochs 40.
RUNS["RELEASE_V2"] = ("unet", ["--seed", "0"])

METRICS = {  # csv column -> regex on the 03 report (SOMA, LBS)
    "mpme_obs":  r"MPME-obs\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
    "mpme_all":  r"MPME \(Mean\)\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
    "p90_all":   r"P90 \(90th Percentile\)\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
    "tau15":     r"Tau > 15 mm\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
    "inter_fat": r"Fat Layer \(S -> M\)\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
    "inter_musc": r"Deep Muscle \(M -> B\)\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
    "rough_skin": r"Skin mean \(deg\)\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
    "rough_musc": r"Muscle mean \(deg\)\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
}
# Fat is reported in posed space since 2026-09-30 (inter_fat/vol_fat). Older reports had T-pose fat in the
# main rows (-> *_fat_tpose) and, for a few runs, posed fat in a "Posed fat ..." diagnostic block.
_OLD_POSED_FAT = {"inter_fat": r"Posed fat intersections \(%\)\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)",
                  "vol_fat": r"Posed fat vol change \(%\)\s+\|\s*([\d.naN]+)\s*\|\s*([\d.naN]+)"}
_TPOSE_FAT = {"inter_fat_tpose": r"T-pose fat intersections \(%\)\s+\|\s*([\d.naN]+)",
              "vol_fat_tpose": r"T-pose fat vol change \(%\)\s+\|\s*([\d.naN]+)"}


def parse_report(text):
    row = {}
    for key, rx in METRICS.items():
        hits = re.findall(rx, text)
        if key.startswith("inter_") and len(hits) >= 2:
            # the same labels appear twice: intersections first, then volume change
            row[f"{key}_soma"], row[f"{key}_lbs"] = hits[0]
            vol = key.replace("inter_", "vol_")
            row[f"{vol}_soma"], row[f"{vol}_lbs"] = hits[1]
        elif hits:
            row[f"{key}_soma"], row[f"{key}_lbs"] = hits[0]
    if "CHANGE, POSED" in text:
        for key, rx in _TPOSE_FAT.items():
            hits = re.findall(rx, text)
            if hits:
                row[f"{key}_soma"] = hits[0]
    else:  # old report: main fat rows were T-pose
        for key in ("inter_fat", "vol_fat"):
            for who in ("soma", "lbs"):
                if f"{key}_{who}" in row:
                    row[f"{key}_tpose_{who}"] = row.pop(f"{key}_{who}")
        for key, rx in _OLD_POSED_FAT.items():
            hits = re.findall(rx, text)
            if hits:
                row[f"{key}_soma"], row[f"{key}_lbs"] = hits[0]
    return row


def run(name, arch, extra, subject, epochs):
    tag = f"{name}_e{epochs}" if epochs != EPOCHS else name
    tag_args = ["--subject", subject, "--arch", arch, "--epochs", str(epochs), "--tag", tag] + extra
    t0 = time.time()
    log = f"{subject}_{tag}" if subject != SUBJECT or tag != name else name
    with open(os.path.join(LOG_DIR, f"{log}_train.log"), "w", encoding="utf-8") as f:
        rc = subprocess.run([sys.executable, "01_end_to_end_training.py"] + tag_args, stdout=f, stderr=subprocess.STDOUT).returncode
    t_train = time.time() - t0
    model = f"{subject}_ABLATION_{arch.upper()}_{tag}"
    ckpt = os.path.join("checkpoints", f"{model}_best_data.pth")
    if rc != 0 or not os.path.exists(ckpt):
        print(f"[{log}] TRAINING FAILED (rc={rc}), see {LOG_DIR}/{log}_train.log")
        return None

    eval_log = os.path.join(LOG_DIR, f"{log}_eval.log")
    # 01 and 03 share the K1 defaults (2026-10-07); forward only what the run changed. 03 auto-detects the rigid head
    # and the static canon next to the checkpoint.
    eval_extra = [a for a in ("--no-marker-offset", "--posed-frame", "--no-posed-data", "--forearm-detwist", "--no-forearm-detwist") if a in extra]
    for a in ("--residual-scale", "--rig"):
        if a in extra:
            eval_extra += [a, extra[extra.index(a) + 1]]
    with open(eval_log, "w", encoding="utf-8") as f:
        rc = subprocess.run([sys.executable, "03_evaluate_metrics.py", "--subject", subject, "--arch", arch,
                             "--checkpoint", ckpt] + eval_extra, stdout=f, stderr=subprocess.STDOUT).returncode
    report = open(eval_log, encoding="utf-8", errors="replace").read()
    if rc != 0:
        print(f"[{log}] EVAL FAILED (rc={rc}), see {eval_log}")
        return None

    best_epoch = re.findall(r"best-data model saved \(epoch (\d+)", open(os.path.join(LOG_DIR, f"{log}_train.log"), encoding="utf-8", errors="replace").read())
    for f in glob.glob(os.path.join("checkpoints", f"{model}_epoch_*.pth")):
        os.remove(f)

    row = {"name": name, "subject": subject, "arch": arch, "args": " ".join(extra), "epochs": epochs,
           "best_data_epoch": best_epoch[-1] if best_epoch else "", "train_min": round(t_train / 60, 1)}
    row.update(parse_report(report))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Comma-separated run names (default: all)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--subjects", default=SUBJECT, help="Comma-separated, e.g. S1,S2,S3")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    if a.list:
        for n, (arch, extra) in RUNS.items():
            print(f"{n:28s} {arch:7s} {' '.join(extra)}")
        return

    names = a.only.split(",") if a.only else list(RUNS)
    os.makedirs(LOG_DIR, exist_ok=True)
    done = set()
    if os.path.exists(CSV_PATH):
        done = {(r["name"], r.get("subject") or SUBJECT, str(r["epochs"])) for r in csv.DictReader(open(CSV_PATH, encoding="utf-8"))}

    for subject, name in [(s, n) for s in a.subjects.split(",") for n in names]:
        if (name, subject, str(a.epochs)) in done:
            print(f"[{subject} {name}] already in {CSV_PATH}, skipping")
            continue
        arch, extra = RUNS[name]
        print(f"[{subject} {name}] training {arch} ({a.epochs} ep) {' '.join(extra)} ...", flush=True)
        row = run(name, arch, extra, subject, a.epochs)
        if row is None:
            continue
        rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8"))) if os.path.exists(CSV_PATH) else []
        rows.append(row)
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[{subject} {name}] done: MPME-obs SOMA {row.get('mpme_obs_soma')} vs LBS {row.get('mpme_obs_lbs')} "
              f"(best epoch {row['best_data_epoch']}, {row['train_min']} min)", flush=True)


if __name__ == "__main__":
    main()
