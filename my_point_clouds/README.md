# `my_point_clouds/`

Drop your own `.ply` (or other) point cloud files here to run ground+wall
detection on them — see root [`GETTING_STARTED.md`](../GETTING_STARTED.md).

```bash
python detect_ground_and_walls.py --ply my_point_clouds/your_cloud.ply --model_variant synthetic
```

This folder is gitignored (everything except this README and
`EXTERNAL_SOURCES.md`) — nothing you drop in here gets committed. Not a
required location; `--ply` accepts any path, this is just a clean,
consistent place to keep your own test files separate from the project's
own datasets.

Don't have a point cloud of your own yet? See
[`EXTERNAL_SOURCES.md`](EXTERNAL_SOURCES.md) in this folder for a list of
free, publicly downloadable point cloud datasets (natural terrain and
urban/city scenes) worth trying.
