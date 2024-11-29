from cyber_record.record import Record


def open_record(record_path):
    record = None
    try:
        record = Record(record_path)
    except Exception as e:
        print(f"{e} : {record_path}")
    return record
