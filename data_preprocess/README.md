# Data Preprocessing

Here we detail the downstream physics simulation data and its preprocessing scripts.

## NASA-CRM

Download data from Case 4 - NASA CRM in this [Google Drive](https://drive.google.com/drive/folders/1KhoZiEHlZhGI8omMwHrp2mZRKGiSAydO)

```bash
python data_preprocess/NASACRM_process.py \
  --train_h5 ./nasa_crm/trainingData_NASA-CRM.h5 \
  --test_h5  ./nasa_crm/testData_NASA-CRM.h5 \
  --outdir ./nasa_npys
```

## AirCraft

```bash
python data_preprocess/AirCraft_process.py \
  --hf_repo GeoPT/Downstream_Physics_Simulation \
  --hf_subdir AirCraft \
  --outdir ./aircraft_npys \
  --pattern "*0.h5" \
  --seed 42
```

## DTCHull

```bash
python data_preprocess/DTCHull_process.py \
  --hf_repo GeoPT/Downstream_Physics_Simulation \
  --hf_subdir DTCHull \
  --outdir ./dtchull_npys \
  --i_start 1 \
  --i_end 130
```

## Car-Crash

```bash
python data_preprocess/CarCrash_process.py \
  --hf_repo GeoPT/Downstream_Physics_Simulation \
  --hf_subdir Car_Crash \
  --outdir ./car_crash_npys \
  --train_start_index 0 \
  --test_start_index 97 \
  --shuffle --seed 42
```

## DrivAerML

Download full data from [Huggingface](https://huggingface.co/datasets/neashton/drivaerml) (31TB).

We suggest splitting the surface data into 20 subsets and the volume data into 400 subsets for more efficient data loading, resulting in the following data structure:

```
Surface_data/
├── run_1/
│   ├── boundary_1_points_part0.npy
│   ├── boundary_1_normals_part0.npy
│   ├── boundary_1_pMeanTrim_part0.npy
│   ├── boundary_1_points_part1.npy
│   ├── boundary_1_normals_part1.npy
│   ├── boundary_1_pMeanTrim_part1.npy
│   └── ...
├── run_2/
│   ├── boundary_2_points_part0.npy
│   ├── boundary_2_normals_part0.npy
│   ├── boundary_2_pMeanTrim_part0.npy
│   └── ...
├── ...
├── run_500/
│   └── ...
```

```
Volume_data/
├── run_1/
│   ├── run_1_cell_centers_part0.npy
│   ├── run_1_pMeanTrim_part0.npy
│   ├── run_1_UMeanTrim_part0.npy
│   ├── run_1_cell_centers_part1.npy
│   ├── run_1_pMeanTrim_part1.npy
│   ├── run_1_UMeanTrim_part1.npy
│   └── ...
├── run_2/
│   └── ...
├── ...
├── run_500/
│   └── ...
```

Next, normalize data geometry with:

- X-direction length of 5
- X-Y center as zero
- Z-min as zero
- Front orientation as -X

Check script `DrivAerML_process.py` for details.
