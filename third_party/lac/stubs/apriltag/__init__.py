"""Import stub installed by slam-benching: LAC imports apriltag at module level only for code paths the
benchmark driver does not use (fiducials / learned mono depth / mesh helpers). Using it raises."""


class _Unavailable:
    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _Unavailable(self._name + "." + attr)

    def __call__(self, *args, **kwargs):
        raise RuntimeError("apriltag is stubbed in the slam-benching LAC image; %s is unavailable" % self._name)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return _Unavailable("apriltag." + name)
