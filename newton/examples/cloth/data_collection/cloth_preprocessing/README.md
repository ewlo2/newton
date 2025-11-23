# Cloth Preprocessing and Dataset Preparation

This folder contains the tools and docs for preparing cloth assets for the dataset pipeline.

- Populate per-cloth `metadata.json` with orientation, center offset, downsampling, and cloth parameters
- Preprocess meshes (center + remesh) and compute quadrant indices with optional visualization
- Output a preprocessed OBJ, `quadrant_indices.json`, and updated `metadata.json`

## 📁 Cloth Library Setup

### Directory Structure

```
assets/
├── cloth_type_01/
│   ├── mesh.obj          # or mesh.stl
│   └── metadata.json     # optional initially; created/updated by metadata script
├── cloth_type_02/
│   ├── mesh.obj
│   └── metadata.json
└── ...
```

### metadata.json (example)

```json
{
  "name": "T-shirt Large",
  "category": "garment",
  "properties": {
    "size": "large",
    "material": "cotton"
  }
}
```

The system will:
- Randomly select a cloth mesh per iteration
- Apply random Z-rotation (if `randomize_orientation: true`)
- Use the same mesh across all environments within that iteration (if configured)
- Automatically handle `.obj` and `.stl` formats

## 🧵 Preprocessing workflow (metadata split)

Cloth preparation is a two-step process:

1) Populate metadata only
- Fills or merges defaults into `metadata.json` in each cloth folder, including:
  - `orientation.base_rotation.axis/degrees`
  - `orientation.default_z_rotation_degrees`
  - `center_offset`
  - `downsampling` settings for later remeshing (informational)
  - `cloth_parameters` (density, stiffness, radius, etc.) used at runtime by the environment
- If `source_mesh_file` is missing, it will be inferred from the first `.obj`/`.stl` in the folder.

2) Preprocess assets (center + remesh + quadrants)
- Requires `metadata.json` to already exist with the fields above.
- Produces a centered, remeshed mesh (OBJ) and writes `quadrant_indices.json`.
- Updates `metadata.json` with:
  - `mesh_file` (the new centered+remeshed asset)
  - `preprocessed` flags
  - `quadrant_indices_file` reference
- Optional visualization is available during preprocessing to colorize quadrant samples.

At runtime, the environment assumes meshes are preprocessed and reads cloth parameters and orientation exclusively from `metadata.json` or the run config—there are no internal defaults.

## ▶️ How to run

PowerShell examples:

```powershell
# 1) Populate or merge metadata defaults
python -m newton.examples.cloth.data_collection.cloth_preprocessing.generate_cloth_metadata --assets-root c:/path/to/assets

# Optional: overwrite existing metadata.json entirely with defaults (then edit by hand)
python -m newton.examples.cloth.data_collection.cloth_preprocessing.generate_cloth_metadata --assets-root c:/path/to/assets --overwrite

# 2) Preprocess assets (center + remesh + compute quadrants)
python -m newton.examples.cloth.data_collection.cloth_preprocessing.preprocess_cloth_assets --assets-root c:/path/to/assets

# Optional visualization during preprocessing
python -m newton.examples.cloth.data_collection.cloth_preprocessing.preprocess_cloth_assets --assets-root c:/path/to/assets --visualize
```

## Notes

- The `downsampling.edge_length_percentage` value in metadata is consumed by preprocessing for remeshing.
- STL axes are adjusted (Y/Z swapped) before preprocessing to match the runtime coordinate system.
- `cloth_parameters` are used by the simulation when creating the cloth (no runtime remeshing is performed).
