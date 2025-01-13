from track.data_struct import ObjectBox
from track.object_filter import ObjectFilter

if __name__ == '__main__':
    input_boxes = []
    # convert to [ObjectBox]
    c_boxes = []
    for i in input_boxes:
        c_boxes.append(ObjectBox(i))
    # Filter
    obj_filter = ObjectFilter()
    obj_filter.input_objects(c_boxes)
    filtered_boxes = obj_filter.filtered_objects
