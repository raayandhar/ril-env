# ril-env


ISSUES (post on the ISSUEs tab in the repo):
- need to run an xarm.step that takes input from the desktop and
  communicates it to the NUC
  - need to make spacemouse connect to desktop and run from there
    instead of from NUC
- replay.py is missing (prev. version does not work) for the seperated code
  - need to have overlay and same features as joint replay
- simulation
  - (Edward) work on simulation; everything should be relatively
    agnostic of whether you run sim or real, i.e., should use the same
    class / inherit from base class. Design decision here.
- consider or support alternatives to `zarr` files -> design discussion
- boil down all functionality to simple primitives: `.get_state`
  (return pose + images), `.get_obs`, `.step`, etc; all scripts are
  very messy right now
- better recordings, easy interopability with other codebases/policy implementations
- better code; best practices, CLI, logging and debugging; documentation
- test installation
