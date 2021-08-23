from calendar import timegm
from datetime import datetime
from time import strftime, struct_time, localtime, time, timezone


def define_recent_range(range):
    """Selects the most recent time period"""
    # Get the Start Time
    current_time = time() + timezone
    start_list = list(localtime(current_time - (range + 2 / 24) * 60 * 60 * 24))
    start_list[4] = 0  # Minutes
    start_list[5] = 0  # Seconds
    start_struct = struct_time(start_list)
    # Get the Current Time
    now_list = list(localtime(current_time - 2 * 60 * 60))
    now_list[4] = 0  # Minutes
    now_list[5] = 0  # Seconds
    end_struct = struct_time(now_list)
    return get_time_lists(start_struct, end_struct)


def define_time_range(start, end):
    """Selects a given time range"""
    start_struct = datetime.datetime.strptime(start, '%Y/%m/%d %H:%M')
    end_struct = datetime.datetime.strptime(end, '%Y/%m/%d %H:%M')
    return get_time_lists(start_struct, end_struct)


def get_time_lists(start_struct, end_struct):
    """Packs up the time lists to be delivered"""
    start_time_list = _set_time(start_struct)
    end_time_list = _set_time(end_struct)
    return start_time_list, end_time_list


def _set_time(_input_time):
    try:
        input_time = _input_time.strftime('%Y/%m/%d %H:%M')
        input_time_long = int(_input_time.strftime('%Y%m%d%H%M%S'))
    except AttributeError:
        input_time = strftime('%Y/%m/%d %H:%M', _input_time)
        input_time_long = int(strftime('%Y%m%d%H%M%S', _input_time))
    
    input_time_string = parse_time_string_to_local(str(input_time_long), 2)[0]
    
    return input_time, input_time_long, input_time_string


def parse_time_string_to_local(downloaded_files, which=0, local=True):
    if which == 0:
        time_string = downloaded_files[0][-25:-10]
        year = time_string[:4]
        month = time_string[4:6]
        day = time_string[6:8]
        hour_raw = int(time_string[9:11])
        minute = time_string[11:13]
    elif which == 3:
        time_string = downloaded_files
        split = time_string.split("_")
        # import pdb; pdb.set_trace()
        year = split[3]
        month = split[4]
        day = split[5].split('t')[0]
        hour_raw = split[5].split('t')[1]
        minute = split[6]
    else:
        time_string = downloaded_files
        year = time_string[:4]
        month = time_string[4:6]
        day = time_string[6:8]
        hour_raw = time_string[8:10]
        minute = time_string[10:12]
    
    struct_time = (int(year), int(month), int(day), int(hour_raw), int(minute), 0, 0, 0, -1)
    # print(struct_time)
    
    if local:
        theTime = localtime(timegm(struct_time))
    else:
        theTime = struct_time
    
    new_time_string = strftime("%I:%M%p %m/%d/%Y", theTime).lower()
    if new_time_string[0] == '0':
        new_time_string = new_time_string[1:]
    
    # print(year, month, day, hour, minute)
    # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)
    time_code = strftime("%Y%m%d%I%M%S", theTime)
    
    return new_time_string, time_code