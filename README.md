# ril-env
**ril-env** is an environment management package designed for robotic control, perception and simulation. It provides an API and programmatic interface to manage state-action flow for both real hardware robots and sensors and their simulation counterparts.

Currently, **ril-env** supports record-replay of the xArm 7 robotic arm on a single machine, with the option of camera perception running in a seperate thread. It also supports recording of the xArm 7 robotic arm on a seperate NUC and visual perception on a main desktop.

More features will be added soon; see the issues tab under the repository.

![System Design](system_design.png)

## Installation
**ril-env** is tested on  Ubuntu Linux with Python 3.

To get started, first clone the repository.
```
git clone https://github.com/UCLA-Robot-Intelligence-Lab/ril-env.git
```
To use this repository with the physical xArm 7 robotic arm, you must use the
[xArm-Python-Sdk](https://github.com/xArm-Developer/xArm-Python-SDK)
package. Follow the instructions on that repo and install the package
at the root of this repository.

Create the conda (recommended) environment. Replace `rilenv` your preferred name for the environment. Run this at the root of your repository:
```
conda create -n rilenv python=3.8
conda activate rilenv
conda install --file requirements.txt
```
Finally, set up the package:
```
python setup.py install
pip install -e . # duplicate?
```
## Basic Usage

Ensure that the arm is enabled on the machine you will run the relevant script on.

To run both the perception and xArm 7 robotic arm, you can use the `joint_script.py` script. 

To run perception and the robotic arm seperately, you can either run `xarm_script.py` and `camera_script.py` seperately, or to run them simultaneously, you can run the `record.sh` shell script from the desktop. Currently, the script is **hardcoded** for a specific NUC and desktop.

You can access nearly all of the parameters for the APIs through their respective config dataclasses. Then, when writing or using a script, you simply need to specify what parameters to change and what to change them to when the `config()` object is being created. 
