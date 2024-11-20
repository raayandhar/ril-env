# ril-env

TODO:
- documentation
- fix replay.py (replaying)
- get overlay working
- get selecting specific frames working
- fix and cleanup the classes
- simulation, setup xarm env, etc for SIM
- script stores files into one named directory instead (collate zarrs?)
- get_state -> can't be done for record (separate computers), but
  could possible be done for replay, have to see

TODO
- Add more to this readme
- Add more comments to the code
- Setup a setup.py/setuptools and fix imports
- Camera calibration (?)
- A reset method to get back to home pose
- CLI arguments (https://click.palletsprojects.com/en/8.1.x/)
- get_state() -> return pose + images
- record and replay actions
- more recording metrics; easy to get monitor (graphs for modeling
  trajectories, etc)
