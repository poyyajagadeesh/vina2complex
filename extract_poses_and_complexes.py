#!/usr/bin/env python3
"""
extract_poses_and_complexes.py
================================
Parse an AutoDock Vina (or compatible) multi-pose PDBQT output and a
receptor PDBQT, then produce:

    1.  poses/pose_N.pdbqt   — individual ligand pose files (Vina format)
    2.  poses/pose_N.pdb     — clean PDB per pose (for MD engines / viewers)
    3.  complexes/complex_pose_N.pdb   — receptor + each ligand pose
    4.  complexes/all_poses.pdb        — multi-MODEL PDB (all poses)
    5.  poses_summary.csv              — energies + RMSD table

Works with ANY protein-ligand pair from AutoDock Vina, Smina, ADFR,
or any tool that writes standard multi-MODEL PDBQT output.

Colab-safe: uses parse_known_args() to tolerate injected kernel args.

Usage
-----
    python3 extract_poses_and_complexes.py \\
        -r receptor.pdbqt \\
        -l ligand_out.pdbqt \\
        -o output_dir

    # Auto-detects /content/ (Colab) or current directory
    # All paths are optional — script resolves intelligently

Requirements: Python standard library only (no pip installs needed).
"""

import argparse
import csv
import os
import re
import sys

# ─────────────────────────────────────────────────────────────────────────────
# PDBQT → PDB: strip charge (cols 67-76) + atom_type (cols 77-79) columns
# Standard PDB ends at column 66. Vina PDBQT extends to 79.
# ─────────────────────────────────────────────────────────────────────────────
PDB_COLUMNS = 66  # keep first 66 chars (up to temp factor), drop charge+type


def pdbqt_to_pdb(line: str) -> str:
    """Strip PDBQT-specific columns → standard PDB ATOM/HETATM line."""
    return line[:PDB_COLUMNS].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Known non-standard residues that should be stripped from receptor
# ─────────────────────────────────────────────────────────────────────────────
KNOWN_NONSTANDARD = {
    # AutoDock / Vina ligand residue names commonly seen in complex PDBs
    "UNK", "UNL", "UNX",
    # Common cofactors that lack AMBER/CHARMM parameters
    "HEM", "FAD", "NAD", "NAP", "COA", "SAM",
    # Metal ions (keep if you need them — remove if force field complains)
    "ZN", "MG", "CA", "FE", "MN", "CO", "NI", "CU",
    # Sugars / glycan fragments
    "NAG", "BMA", "MAN", "GAL", "GLC", "FUC", "XYP", "BGC",
}


# ─────────────────────────────────────────────────────────────────────────────
# Receptor PDBQT → clean PDB
# ─────────────────────────────────────────────────────────────────────────────
# Map of non-standard residue names → closest standard equivalent
NONSTANDARD_REPLACEMENTS = {
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
    "ASH": "ASP", "GLH": "GLU", "CYX": "CYS", "CYM": "CYS",
    "LYN": "LYS", "TYR": "TYR", "TRP": "TRP",
}


def process_receptor(pdbqt_path: str, out_dir: str) -> str:
    """
    Read receptor PDBQT, strip PDBQT columns, rename non-standard residues,
    write clean PDB. Returns path to the clean PDB.
    """
    base = os.path.splitext(os.path.basename(pdbqt_path))[0]
    out_path = os.path.join(out_dir, f"{base}.pdb")

    with open(pdbqt_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                resname = line[17:20].strip()

                # Skip known non-standard / ligand residues entirely
                if resname in KNOWN_NONSTANDARD:
                    continue

                # Rename common HIS variants to standard HIS
                if resname in NONSTANDARD_REPLACEMENTS:
                    line = line[:17] + NONSTANDARD_REPLACEMENTS[resname] + line[20:]

                fout.write(pdbqt_to_pdb(line) + "\n")

            elif line.startswith(("TER", "END", "REMARK", "HEADER", "TITLE", "CONECT")):
                fout.write(line)

    print(f" Receptor PDB: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Ligand PDBQT → individual poses
# ─────────────────────────────────────────────────────────────────────────────
def extract_ligand_poses(ligand_pdbqt_path: str, out_dir: str) -> list:
    """
    Parse multi-MODEL Vina-compatible PDBQT and extract each pose.

    Handles:
    - AutoDock Vina (ROOT/BRANCH/ENDBRANCH tree)
    - Smina / Vina-GPU (same MODEL-based format)
    - Any tool writing standard multi-MODEL PDBQT with HETATM records

    Returns list of dicts:
        [{"pose": 1, "energy": -7.3, "rmsd_lb": 0.0, "rmsd_ub": 0.0,
          "pdbqt": "...", "pdb": "...", "n_atoms": 29}, ...]
    """
    base = os.path.splitext(os.path.basename(ligand_pdbqt_path))[0]
    poses_dir = os.path.join(out_dir, "poses")
    complexes_dir = os.path.join(out_dir, "complexes")
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(complexes_dir, exist_ok=True)

    with open(ligand_pdbqt_path, "r") as f:
        content = f.read()

    # ── Split by MODEL ────────────────────────────────────────────────────
    # Handles "MODEL 1" through "MODEL 9" (standard multi-pose PDBQT)
    model_blocks = re.split(r"\nMODEL\s+\d+", content)
    model_blocks = [b for b in model_blocks if "HETATM" in b]

    if not model_blocks:
        raise ValueError(
            "No MODEL blocks found in ligand PDBQT. "
            "Expected multi-pose Vina/Smina output with MODEL records."
        )

    n_poses = len(model_blocks)
    print(f"  Found {n_poses} pose(s) in ligand file.")

    # ── Extract Vina-compatible scores ────────────────────────────────────
    # Supports both:
    #   REMARK VINA RESULT:  -7.3  0.000  0.000   (Vina standard)
    #   REMARK VINA RESULT:  -7.3  0.000  0.000   (Smina compatible)
    vina_remarks = re.findall(
        r"REMARK\s+VINA\s+RESULT:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
        content,
    )

    results = []
    for i, block in enumerate(model_blocks, 1):
        # Parse docking score
        energy, rmsd_lb, rmsd_ub = None, None, None
        if i <= len(vina_remarks):
            energy = float(vina_remarks[i - 1][0])
            rmsd_lb = float(vina_remarks[i - 1][1])
            rmsd_ub = float(vina_remarks[i - 1][2])

        # ── Extract HETATM lines, renumber atom serial → 1..N ───────────
        pdbqt_block = block.strip()
        if not pdbqt_block.endswith("ENDMDL"):
            pdbqt_block += "\nENDMDL"

        hetatm_lines = re.findall(r"^(HETATM\s+\d+.*)$", pdbqt_block, re.MULTILINE)

        clean_hetatm = []
        for serial, line in enumerate(hetatm_lines, 1):
            new_line = f"HETATM{serial:>5}" + line[11:]
            clean_hetatm.append(new_line)

        # ── Write pose PDBQT (preserves ROOT/BRANCH tree) ────────────────
        pdbqt_out = os.path.join(poses_dir, f"pose_{i}.pdbqt")
        with open(pdbqt_out, "w") as f:
            f.write(f"MODEL  {i}\n")
            if energy is not None:
                f.write(
                    f"REMARK VINA RESULT:     {energy:.1f}  {rmsd_lb:.3f}  {rmsd_ub:.3f}\n"
                )
            f.write(pdbqt_block)
            if not pdbqt_block.endswith("ENDMDL"):
                f.write("\nENDMDL")

        # ── Write pose as clean PDB ──────────────────────────────────────
        pdb_out = os.path.join(poses_dir, f"pose_{i}.pdb")
        with open(pdb_out, "w") as f:
            if energy is not None:
                f.write(f"REMARK Pose {i} | Energy: {energy:.1f} kcal/mol\n")
                f.write(f"REMARK RMSD from best: {rmsd_lb:.3f} Å\n")
            for line in clean_hetatm:
                f.write(pdbqt_to_pdb(line) + "\n")
            f.write("TER\nEND\n")

        score_str = f"{energy:.1f} kcal/mol" if energy is not None else "N/A"
        print(f"  Pose {i}: energy={score_str}  atoms={len(clean_hetatm)}")

        results.append({
            "pose": i,
            "energy": energy,
            "rmsd_lb": rmsd_lb,
            "rmsd_ub": rmsd_ub,
            "n_atoms": len(clean_hetatm),
            "pdbqt": pdbqt_out,
            "pdb": pdb_out,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Complex PDBs: receptor + ligand pose
# ─────────────────────────────────────────────────────────────────────────────
def build_complexes(receptor_pdb: str, poses: list, out_dir: str):
    """
    For each pose, concatenate receptor PDB + ligand PDB into a complex PDB.
    Also writes a multi-MODEL PDB with all poses.
    """
    complexes_dir = os.path.join(out_dir, "complexes")
    os.makedirs(complexes_dir, exist_ok=True)

    with open(receptor_pdb, "r") as f:
        receptor_lines = f.read()

    for p in poses:
        # Per-pose complex
        complex_path = os.path.join(complexes_dir, f"complex_pose_{p['pose']}.pdb")
        with open(complex_path, "w") as f:
            if p["energy"] is not None:
                f.write(
                    f"REMARK Complex — pose {p['pose']} | "
                    f"Energy: {p['energy']:.1f} kcal/mol\n"
                )
            f.write(receptor_lines)
            f.write("\n")
            with open(p["pdb"], "r") as lf:
                for line in lf:
                    if line.startswith(("HETATM", "ATOM")):
                        f.write(line)
            f.write("TER\nEND\n")
        print(f"   Complex pose {p['pose']}: {complex_path}")

    # Multi-model PDB (all poses in one file)
    multi_model_path = os.path.join(complexes_dir, "all_poses.pdb")
    with open(multi_model_path, "w") as f:
        for p in poses:
            f.write(f"MODEL    {p['pose']}\n")
            f.write(receptor_lines)
            f.write("\n")
            with open(p["pdb"], "r") as lf:
                for line in lf:
                    if line.startswith(("HETATM", "ATOM")):
                        f.write(line)
            f.write("TER\nENDMDL\n")
    print(f"   Multi-model PDB: {multi_model_path}")

    # Summary CSV
    csv_path = os.path.join(out_dir, "poses_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Pose", "Energy_kcal/mol", "RMSD_lowerbound_Å",
            "RMSD_upperbound_Å", "N_atoms",
            "Ligand_PDBQT", "Ligand_PDB", "Complex_PDB",
        ])
        for p in poses:
            writer.writerow([
                p["pose"],
                f"{p['energy']:.1f}" if p["energy"] is not None else "N/A",
                f"{p['rmsd_lb']:.3f}" if p["rmsd_lb"] is not None else "N/A",
                f"{p['rmsd_ub']:.3f}" if p["rmsd_ub"] is not None else "N/A",
                p["n_atoms"],
                os.path.basename(p["pdbqt"]),
                os.path.basename(p["pdb"]),
                f"complex_pose_{p['pose']}.pdb",
            ])
    print(f"   Summary CSV: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Extract Vina/Smina/ADFR ligand poses and build "
            "protein-ligand complexes (works for any protein-ligand pair)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 extract_poses_and_complexes.py -r 4H6J.pdbqt -l out.pdbqt\n"
            "  python3 extract_poses_and_complexes.py -r rec.pdbqt -l lig.pdbqt -o results\n"
            "\n"
            "Upload both files to /content/ (Colab) or run from the same directory."
        ),
    )
    ap.add_argument(
        "-r", "--receptor", required=True,
        help="Receptor PDBQT file (any protein PDBQT)",
    )
    ap.add_argument(
        "-l", "--ligand", required=True,
        help="Ligand docking output PDBQT (multi-MODEL, Vina/Smina/ADFR format)",
    )
    ap.add_argument(
        "-o", "--out_dir", default=None,
        help="Output directory (default: <ligand_basename>_analysis)",
    )
    args = ap.parse_known_args()[0]  # Colab-safe

    # Resolve paths: check exact path first, then /content/ (Colab), then cwd
    search_dirs = ["/content", "."]

    def resolve(name: str) -> str:
        if os.path.isfile(name):
            return name
        for d in search_dirs:
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                return candidate
        return name

    receptor_path = resolve(args.receptor)
    ligand_path = resolve(args.ligand)

    if not os.path.isfile(receptor_path):
        sys.exit(f" Receptor not found: {args.receptor}")
    if not os.path.isfile(ligand_path):
        sys.exit(f" Ligand not found: {args.ligand}")

    # Output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        lig_base = os.path.splitext(os.path.basename(ligand_path))[0]
        out_dir = f"{lig_base}_analysis"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "poses"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "complexes"), exist_ok=True)

    print("=" * 60)
    print("  Vina Pose Extraction & Complex Builder")
    print("=" * 60)
    print(f"  Receptor : {receptor_path}")
    print(f"  Ligand   : {ligand_path}")
    print(f"  Output   : {out_dir}/")
    print("=" * 60)

    # Step 1: Process receptor
    print("\n Processing receptor …")
    receptor_pdb = process_receptor(receptor_path, out_dir)

    # Step 2: Extract ligand poses
    print("\n Extracting ligand poses …")
    poses = extract_ligand_poses(ligand_path, out_dir)

    # Step 3: Build complexes
    print("\n Building protein-ligand complexes …")
    build_complexes(receptor_pdb, poses, out_dir)

    # Summary table
    print("\n" + "=" * 60)
    print("  All poses (sorted by energy)")
    print("=" * 60)
    sorted_poses = sorted(
        [p for p in poses if p["energy"] is not None],
        key=lambda p: p["energy"],
    )
    if sorted_poses:
        print(f"  {'#':>3}  {'Energy':>8}  {'RMSDlb':>8}  {'RMSDub':>8}  {'Atoms':>6}")
        print(f"  {'─' * 3}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 6}")
        for p in sorted_poses:
            print(
                f"  {p['pose']:>3}  {p['energy']:>8.1f}  "
                f"{p['rmsd_lb']:>8.3f}  {p['rmsd_ub']:>8.3f}  "
                f"{p['n_atoms']:>6}"
            )
    else:
        print("  (no energy scores found in file)")

    # Show poses without scores too
    no_score = [p for p in poses if p["energy"] is None]
    if no_score:
        print(f"\n  {len(no_score)} pose(s) without Vina scores (file may use different format)")

    print("=" * 60)
    print(f"\n  All outputs in: {os.path.abspath(out_dir)}/")
    print(f"   ├── poses/       individual ligand poses (pdb + pdbqt)")
    print(f"   └── complexes/   protein-ligand complexes + all_poses.pdb")


if __name__ == "__main__":
    main()
