from calendar import timegm
import datetime
from time import strftime, localtime, time, timezone
from astropy import units as u

# Named constants for time string parsing
YEAR_SLICE, MONTH_SLICE, DAY_SLICE = slice(0, 4), slice(4, 6), slice(6, 8)
HOUR_SLICE, MINUTE_SLICE, SECOND_SLICE = slice(8, 10), slice(10, 12), slice(12, 14)


def define_recent_range(range_days, offset=0):
    """Selects the most recent time period"""
    current_time = time() + timezone
    start_time = current_time - ((range_days + 2 / 24) * 60 * 60 * 24)
    end_time = current_time - 2 * 60 * 60

    return get_time_lists(localtime(start_time), localtime(end_time))


def define_time_range(start, end):
    """Selects a given time range"""
    start_struct = datetime.datetime.strptime(start, "%Y/%m/%d %H:%M:%S")
    end_struct = datetime.datetime.strptime(end, "%Y/%m/%d %H:%M:%S")
    return get_time_lists(start_struct, end_struct)


def get_time_lists(start_struct, end_struct):
    """Packs up the time lists to be delivered"""
    return _set_time(start_struct), _set_time(end_struct)


def _set_time(_input_time):
    try:
        input_time = _input_time.strftime("%Y/%m/%d %H:%M:%S")
        input_time_long = int(_input_time.strftime("%Y%m%d%H%M%S"))
    except AttributeError:
        input_time = strftime("%Y/%m/%d %H:%M:%S", _input_time)
        input_time_long = int(strftime("%Y%m%d%H%M%S", _input_time))

    input_time_string = parse_time_string_to_local(str(input_time_long), 2)[0]

    return input_time, input_time_long, input_time_string


def parse_time_string_to_local(input_string, which=0, local=True):
    from astropy.time import Time
    if isinstance(input_string, Time):
        input_string = input_string.iso

    if which == 0:
        time_string = input_string[0][-25:-10]
    elif which == 3:
        time_string = input_string.split("_")[2]
    elif which == 2:
        time_string = input_string
    else:
        time_string = (
            str(input_string).split(" ")[0]
            .replace("-", "")
            .replace(":", "")
            .replace("T", "")
        )

    # Rest of your function remains the same

    year, month, day = (
        time_string[YEAR_SLICE],
        time_string[MONTH_SLICE],
        time_string[DAY_SLICE],
    )
    hour, minute, second = (
        time_string[HOUR_SLICE],
        time_string[MINUTE_SLICE],
        time_string[SECOND_SLICE],
    )
    if not hour:
        hour = "00"
    if not minute:
        minute = "00"
    if not second:
        second = "00"

    struct_time = (
        int(year),
        int(month),
        int(day),
        int(hour),
        int(minute),
        int(second),
        0,
        0,
        -1,
    )

    theTime = localtime(timegm(struct_time)) if local else struct_time
    new_time_string = strftime("%I:%M:%S%p %m/%d/%Y", theTime).lower()
    time_code = strftime("%Y%m%d%I%M%S", theTime)

    return new_time_string, time_code
