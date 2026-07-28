# AutoDock Vina Pose Extractor

A simple Python script to extract individual ligand poses from an AutoDock Vina multi-pose PDBQT file and generate protein–ligand complex PDB files using a receptor PDBQT.

## Features

- Extracts each docking pose into separate ligand PDBQT files
- Converts ligand poses to standard PDB format
- Converts receptor PDBQT to PDB
- Generates one protein–ligand complex PDB for each docking pose
- Creates a multi-model PDB containing all complexes
- Exports a CSV summary of docking scores and RMSD values

## Requirements

- Python 3.8 or later
- No external Python packages required

## Usage

```bash
python extract_poses_and_complexes.py \
    -r receptor.pdbqt \
    -l ligand_out.pdbqt
```

Or specify an output directory:

```bash
python extract_poses_and_complexes.py \
    -r receptor.pdbqt \
    -l ligand_out.pdbqt \
    -o results
```

## Output

```
output/
├── poses/
│   ├── pose_1.pdbqt
│   ├── pose_1.pdb
│   ├── pose_2.pdbqt
│   └── ...
├── complexes/
│   ├── complex_pose_1.pdb
│   ├── complex_pose_2.pdb
│   ├── ...
│   └── all_poses.pdb
└── poses_summary.csv
```

## License

MIT License
