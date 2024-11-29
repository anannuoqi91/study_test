class Selector:
    def __init__(self) -> None:
        self._now_idx = 0
        self._enable_select_measure = False
        self._last_selected_idx = 0
        self._last_selected_timestamp = 0
        self._enable_check_error_measure = True

    def set_current_idx(self, idx):
        self._now_idx = idx

    def set_current_measure(self, measure_list):
        if not self._enable_select_measure:
            self._vec_selected_measure = measure_list.copy()
            self._vec_wasted_measure = []
            return
        self._vec_selected_measure = []
        self._vec_wasted_measure = []
        for measure in measure_list:
            if self._enable_check_error_measure and not self.check_error_measure(measure):
                self._vec_wasted_measure.append(measure)
                continue
            self._vec_selected_measure.append(measure)

    def check_error_measure(self, measure):
        if measure.timestamp == 0:
            return False

        if (self._now_idx < self._last_selected_idx):
            self._last_selected_timestamp = 0
            self._last_selected_idx = 0

        if self._last_selected_timestamp != 0 and \
                (measure.timestamp - self._last_selected_timestamp) / 1e9 > (self._now_idx - self._last_selected_idx) * 1 * 2:
            return False

        if (measure.box_l <= 0 or measure.box_w <= 0 or measure.box_h() <= 0):
            return False

        return True

    @property
    def selected_measure(self):
        return self._vec_selected_measure.copy()
