Below is a concrete, implementation-ready architecture for **reciprocal Green’s-function (GF) generation** with both **SPECFEM3D** and **OpenSWPC**.

A key framing difference:

* **OpenSWPC** has an explicit reciprocity mode (`green_mode`) that directly outputs MT/body-force responses for many virtual source points in one run. ([openswpc.github.io][1])
* **SPECFEM3D** does not expose that as a single “green_mode” switch, but its adjoint/source-inversion workflow gives the same building block: run with virtual sources at receivers, then read source-location strain/displacement outputs. ([specfem3d.readthedocs.io][2])

---

## 1) SPECFEM3D-based reciprocal GF architecture

### 1.1 Directory schema

```text
gf_specfem/
  00_model/
    mesh/                              # mesh + material model (3D velocity model)
    DATA/
      Par_file.base
      STATIONS.template
    DATABASES_MPI/                     # from xgenerate_databases (cached)
    model_manifest.json                # hash, bounds, dx_target, dt, version
  01_inventory/
    receivers.csv                       # station_id, net, lon, lat, elev, burial
    source_nodes.parquet                # src_id, x,y,z, lon,lat, depth, chunk_id
    source_chunks/
      chunk_0000.csv
      chunk_0001.csv
  02_runs/
    STA_<station_id>/
      CMP_X/
        chunk_0000/
          DATA/{Par_file,STATIONS,STATIONS_ADJOINT,CMTSOLUTION}
          SEM/                         # adjoint source files NT.STA.BX?.adj
          LOCAL_PATH/
          OUTPUT_FILES/
        chunk_0001/
      CMP_Y/
      CMP_Z/
  03_ingest/
    parsers/
      parse_sem_strain.py
      build_gf_zarr.py
  04_gf_store/
    model_<model_hash>/
      gf_tensor.zarr/                  # chunked tensor store
      index.parquet
  05_qc/
    travel_time_checks/
    reciprocity_spot_checks/
    logs/
  06_db/
    gfmeta.sqlite                      # or Postgres
```

### 1.2 How reciprocity is implemented in SPECFEM3D

1. Use adjoint/source-inversion mode: virtual sources at receiver locations, gradient info collected at source locations. ([specfem3d.readthedocs.io][2])
2. For each `(station, component)` run, provide the three component files (`BXE/BXN/BXZ`) even if only one is non-zero (SPECFEM expectation). ([specfem3d.readthedocs.io][2])
3. At completion, collect source-location strain outputs (`SNN, SEE, SZZ, SNE, SNZ, SEZ`) and (optionally) displacement outputs. These six strain traces are exactly what you need as MT elementary basis at that source node for that receiver-component excitation. ([specfem3d.readthedocs.io][2])
4. Put many candidate source nodes in one run by concatenating multiple entries in `CMTSOLUTION` (chunk this if file gets too large). ([specfem3d.readthedocs.io][3])

### 1.3 Database fields (canonical schema, shared by both engines)

```sql
-- model-level metadata
CREATE TABLE model (
  model_id TEXT PRIMARY KEY,           -- UUID
  engine TEXT NOT NULL,                -- 'specfem3d' | 'openswpc'
  model_hash TEXT NOT NULL,            -- SHA256 of velocity+mesh+params
  grid_type TEXT,                      -- SEM mesh / FDM Cartesian
  dt REAL NOT NULL,
  nt INTEGER NOT NULL,
  fmax_target REAL,
  units TEXT,
  created_at TEXT NOT NULL,
  engine_version TEXT NOT NULL,
  config_json TEXT NOT NULL
);

CREATE TABLE receiver (
  receiver_id TEXT PRIMARY KEY,        -- station code unique key
  network TEXT,
  station TEXT,
  x REAL, y REAL, z REAL,
  lon REAL, lat REAL, depth REAL,
  component_frame TEXT NOT NULL,       -- ENZ / XYZ / NED
  is_active INTEGER NOT NULL
);

CREATE TABLE source_node (
  source_id INTEGER PRIMARY KEY,
  chunk_id INTEGER NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
  lon REAL, lat REAL, depth REAL,
  inside_model INTEGER NOT NULL,
  tetra_id INTEGER                      -- for interpolation lookup
);

CREATE TABLE gf_trace (
  trace_id TEXT PRIMARY KEY,            -- UUID
  model_id TEXT NOT NULL,
  receiver_id TEXT NOT NULL,
  rec_component TEXT NOT NULL,          -- X|Y|Z (or N|E|Z)
  source_id INTEGER NOT NULL,
  basis TEXT NOT NULL,                  -- Mxx,Myy,Mzz,Mxy,Mxz,Myz (+Fx,Fy,Fz optional)
  dt REAL NOT NULL,
  nt INTEGER NOT NULL,
  t0 REAL NOT NULL,
  amplitude_unit TEXT NOT NULL,
  vertical_sign TEXT NOT NULL,          -- store convention explicitly
  file_uri TEXT NOT NULL,               -- zarr path/chunk
  checksum TEXT NOT NULL,
  qc_peak_snr REAL,
  qc_flag TEXT,
  FOREIGN KEY(model_id) REFERENCES model(model_id),
  FOREIGN KEY(receiver_id) REFERENCES receiver(receiver_id),
  FOREIGN KEY(source_id) REFERENCES source_node(source_id)
);

CREATE TABLE job_run (
  job_id TEXT PRIMARY KEY,
  model_id TEXT NOT NULL,
  receiver_id TEXT NOT NULL,
  rec_component TEXT NOT NULL,
  chunk_id INTEGER NOT NULL,
  nproc INTEGER,
  status TEXT NOT NULL,                 -- queued/running/success/failed
  wall_seconds REAL,
  stdout_path TEXT,
  stderr_path TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
);
```

### 1.4 Interpolation strategy (SPECFEM store)

Use a **3-stage interpolation/composition**:

1. **Spatial interpolation in source space (3D)**

   * Build tetrahedralization over `source_nodes` (or regular-cell trilinear if regular).
   * For event source (x_s), find enclosing tetrahedron and barycentric weights (w_1...w_4).
   * Interpolate each elementary basis trace sample-wise:
     [
     g_b(t, x_s) = \sum_{k=1}^4 w_k, g_b(t, x_k)
     ]
2. **Moment-tensor combination (exact linear step)**
   Waveform is linear in MT components, so combine six basis traces directly:
   [
   u_i(t)=\sum_{b\in{Mxx,\dots,Myz}} m_b, g_{i,b}(t,x_s)
   ]
   The linearity is exact for MT parameters. 
3. **Origin-time fractional shift**
   Apply FFT phase shift or Lagrange fractional-delay filter for sub-sample (t_0).

**Practical fallback:** if source exits convex hull, use nearest-neighbor + penalty flag (do not silently extrapolate).

### 1.5 Parallelization plan (SPECFEM)

* **Embarrassingly parallel axis:** `(receiver, component)` i.e., `3 * Nr` jobs.
* **Nested axis:** `source_chunk` (if too many CMTSOLUTION entries per job).
* **Intra-job parallelism:** MPI ranks per SPECFEM solve (plus GPU build if available). SPECFEM is MPI-based and has GPU pathways. ([specfem3d.readthedocs.io][4])
* **Important caching rule:** generate databases once; only a limited subset of `Par_file` fields can be changed afterward without rerunning `xgenerate_databases`. ([specfem3d.readthedocs.io][3])

Recommended scheduler layout:

* Job array index maps to `(station_id, component, chunk_id)`.
* Keep one immutable `DATABASES_MPI` per velocity model hash.
* Ingest traces to Zarr immediately after each successful job.

---

## 2) OpenSWPC-based reciprocal GF architecture

### 2.1 Directory schema

```text
gf_openswpc/
  00_model/
    velocity_model/                     # 3D model files for OpenSWPC
    params/base.par
    model_manifest.json
  01_inventory/
    stations.lst                        # includes virtual receiver stations
    glst_chunks/
      glst_0000.txt                     # x y z gid  (or lon lat z gid)
      glst_0001.txt
  02_runs/
    STA_<station_id>/
      CMP_X/
        chunk_0000/
          input.par
          odir/
            green/
              <gid>/
                <title>__x__mxx__.sac
                ...
      CMP_Y/
      CMP_Z/
  03_ingest/
    parse_green_sac.py
    build_gf_zarr.py
  04_gf_store/
    model_<model_hash>/
      gf_tensor.zarr
      index.parquet
  05_qc/
  06_db/
    gfmeta.sqlite
```

### 2.2 Built-in reciprocity workflow in OpenSWPC

For each run set:

* `green_mode = .true.`
* `green_stnm = <station name in station list>`
* `green_cmp in {x,y,z}` (3 runs needed for full component response)
* `fn_glst = <virtual source list file>`
* optional `green_maxdist` to prune distant virtual sources
* optional `green_bforce=.true.` if you also want body-force response. ([openswpc.github.io][1])

Outputs are written per source `gid` under:
`(odir)/green/(gid)` with names like `(title)__(green_cmp)__mij__.sac` (and `fi` for body force). ([openswpc.github.io][1])

### 2.3 Output handling details that matter for database design

* OpenSWPC waveform output supports `sac`, `tar_st`, `tar_node`, and `csf`; `csf` is deprecated in newer docs, so prefer tar modes for huge runs. ([openswpc.github.io][5])
* File naming conventions are explicit (`(odir)/wav/...`) and should be mirrored into your indexer. ([openswpc.github.io][5])
* Coordinate/parallel setup uses Cartesian domain with 2D MPI partitioning (`nproc_x`, `nproc_y`) for 3D runs. ([openswpc.github.io][6])

### 2.4 Interpolation strategy (OpenSWPC store)

Use the same three-stage strategy as above (source-space interpolation, exact MT linear combination, fractional time shift), so your inversion code is engine-agnostic.

Two OpenSWPC-specific notes:

1. OpenSWPC already interpolates input velocity model fields onto computational grid (bicubic step in mapping workflow), so keep a **model_hash** that includes both original model and OpenSWPC grid params. ([openswpc.github.io][6])
2. Respect wavelength/stability rules when setting `dx,dt` for high frequency; otherwise interpolation quality won’t rescue dispersive synthetics. ([openswpc.github.io][6])

### 2.5 Parallelization plan (OpenSWPC)

* **Outer axis:** `(station, component)` = `3 * Nr`
* **Second axis:** `glst_chunk`
* **Intra-job:** OpenSWPC MPI domain decomposition via `nproc_x * nproc_y`. ([openswpc.github.io][6])
* Use `green_maxdist` aggressively to avoid unnecessary source computations. ([openswpc.github.io][1])

Operational caveat: checkpoint/restart feature is removed in v24.09 docs; if you rely on restart, either chunk runs more aggressively or pin to an older supported version as documented. ([openswpc.github.io][7])

---

## 3) Suggested interpolation/QC defaults (for both engines)

* **Source grid spacing:** choose by target highest reliable frequency and minimum (V_s), with ≥8–10 points per shortest wavelength as conservative high-frequency target (OpenSWPC docs mention 5–10 minimum). ([openswpc.github.io][6])
* **Two-level library:** coarse global GF grid + automatic local refinement around best-fit region.
* **QC metrics per trace:** peak amplitude sanity, first-arrival monotonicity vs distance, spectral slope, reciprocity spot-check residual.
* **Sign/frame normalization:** persist vertical sign and component frame in DB; OpenSWPC notes special vertical convention handling in reciprocity output. ([openswpc.github.io][1])

---

## 4) Which one to prioritize for your current need

Given your goal (fast repeated MT inversions after precompute):

* **Fastest path to reciprocal GF library:** **OpenSWPC**, because reciprocity is first-class and directly emits MT response traces per virtual source list. ([openswpc.github.io][1])
* **Potentially higher-fidelity SEM path in complex media:** **SPECFEM3D reciprocal-adjoint pipeline**, but with heavier orchestration and larger run management overhead. SPECFEM-based reciprocal MT workflows are used in induced-seismicity contexts and rely on 18 elementary seismograms + reciprocity. 

---.

[1]: https://openswpc.github.io/2._Parameters/0210_reciprocity/ "Reciprocity Mode - OpenSWPC"
[2]: https://specfem3d.readthedocs.io/en/latest/07_adjoint_simulations "07 Adjoint Simulations - SPECFEM3D_Cartesian"
[3]: https://specfem3d.readthedocs.io/en/latest/05_running_the_solver "05 Running The Solver - SPECFEM3D_Cartesian"
[4]: https://specfem3d.readthedocs.io/en/latest/01_introduction "01 Introduction - SPECFEM3D_Cartesian"
[5]: https://openswpc.github.io/2._Parameters/0205_output/ "Simulation Data Output - OpenSWPC"
[6]: https://openswpc.github.io/2._Parameters/0203_coord/ "Coordinates and Parallel Computation - OpenSWPC"
[7]: https://openswpc.github.io/2._Parameters/0209_checkpoint/ "Checkpointing and Restarting - OpenSWPC"
