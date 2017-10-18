import sys


#velikost v byte
#original ->  https://gist.github.com/bosswissam/a369b7a31d9dcab46b4a034be7d263b2
def get_size(object, seen=None):

    size = sys.getsizeof(object)

    if seen is None:
        seen = set()

    object_id = id(object)

    if object_id in seen:
        return 0


    seen.add(object_id)

    if isinstance(object, dict):
        size += sum([get_size(v, seen) for v in object.values()])
        size += sum([get_size(k, seen) for k in object.keys()])
    elif hasattr(object, '__dict__'):
        size += get_size(object.__dict__, seen)
    elif hasattr(object, '__iter__') and not isinstance(object, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in object])

    return size





def boolAnoNe(bool):
    if bool == True:
        return "ANO"
    elif bool == False:
        return "NE"
    return