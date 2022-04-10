"""Set up SlowFast Environment."""
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.



from iopath.common.file_io import PathManagerFactory

_ENV_SETUP_DONE = False
pathmgr = PathManagerFactory.get(key="pyslowfast")
checkpoint_pathmgr = PathManagerFactory.get(key="pyslowfast_checkpoint")


def setup_environment():
    """Set up Environment."""
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True