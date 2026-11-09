# Experiment configs

One YAML per experiment; run with `slambench run configs/<subdir>/<name>.yaml` (or via
`scripts/slurm_slambench.sbatch`). The `name:` field, not the file path, names the output directory
`runs/<name>/<run id>/`, so files can move without breaking `load_map:` / `map_run:` references.

| folder | contents |
|---|---|
| `lunapolaris/` | real LunaPolaris single-session runs, including map-reuse session A (`*_map*`, `save_map: true`) |
| `lunapolaris/map_reuse/` | LunaPolaris map-reuse session B (`lunapolaris_mapreuse_map<A>_loc<B>_*`) |
| `omnilrs/lunalab/` | OmniLRS Lunalab replays of LunaPolaris trajectories (`omnilrs_seq*`) |
| `omnilrs/lunaryard/` | OmniLRS Lunaryard aerial survey under four lightings, incl. session A maps |
| `omnilrs/largescale/` | OmniLRS LargeScale (LOLA Site20) aerial runs, incl. session A maps |
| `omnilrs/map_reuse/` | OmniLRS map-reuse session B: Lunaryard (`omnilrs_mapreuse_*`) and LargeScale (`omnilrs_lsmapreuse_*`) |
| `tumvi/` | estimator installation check on TUM-VI |

`batch_example.yaml` and `rejection_examples.yaml` are batch files (`slambench batch`); their
`experiments:` paths are relative to this directory.

Names follow `<dataset>_<sequence or scenario>_<estimator>_<mode>[_<variant>]`, e.g.
`lunapolaris_seq01_orbslam3_stereo_lean`. Session B configs are
`<dataset>_mapreuse_map<A>_<query>_<estimator>_<reuse mode or variant>`. Configs whose inputs were
deleted but whose numbers a results page reports start with a `# HISTORICAL:` comment.
