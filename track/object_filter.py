class ObjectFilter:
    def __init__(self) -> None:
        self._out_objects = None

    def input_objects(self, objects: list):
        self._out_objects = objects
        self._filter_objects()

    def _filter_objects(self):
        pass

    @property
    def filtered_objects(self):
        return self._out_objects
