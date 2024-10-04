# ril-env
To use this repository you must use the
[xArm-Python-Sdk](https://github.com/xArm-Developer/xArm-Python-SDK)
package. Follow the instructions on that repo and install the package
at the root of this repository.

See `example.py` for an example of how to use this repository.

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
