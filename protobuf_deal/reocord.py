from cyber_record.record import Record
from protobuf_deal.write_pcd import write_pcd_ietsfss


class ReocrdRW:
    def __init__(self, path) -> None:
        self._path = path
        self._record = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close_record()

    def open_record(self):
        """
        Opens a record file and returns a Record object.

        This function attempts to create a Record object from the given file path.
        If successful, it returns the Record object. If an exception occurs during
        the process, it prints the error message along with the record path.

        Args:
            record_path (str): The file path of the record to be opened.

        Returns:
            Record or None: A Record object if the file is successfully opened,
                            or None if an exception occurs.
        """
        try:
            self._record = Record(self._path)
        except Exception as e:
            print(f"{e} : {self._path}")

    def close_record(self):
        self._record.close()

    @property
    def messages(self):
        return self._record.read_messages()

    def messagepc2_to_pcd(self, message, file_path):
        pts = []
        for pt in message.point:
            pts.append([pt.x, pt.y, pt.z, pt.intensity,
                       pt.elongation, pt.timestamp, pt.sub_id, pt.flags, pt.scan_id, pt.scan_idx])
        if pts:
            write_pcd_ietsfss(file_path, pts)


if __name__ == "__main__":
    record_path = '/home/demo/Documents/datasets/1010s/03.1.00000'
    with ReocrdRW(record_path) as rw:
        rw.open_record()
        for topic, message, t in rw.messages:
            if topic == 'omnisense/distortion_correction/01/PointCloud':
                pc2 = rw.messagepc2_to_pcd(message, f'{t}_pc2.pcd')
