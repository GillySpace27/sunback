from WebExecute import WebExecute
import sys
import sunback as sb


# # Main Command Structure
def start(params):
    """Select whether to run or to debug"""
    __print_header(params)
    
    if params.is_debug():
        __debug_mode(params)
    else:
        __run_mode(params)


def __print_header(params):
    print("\nSunback: Live SDO Background Updater \nWritten by Chris R. Gilly")
    print("Check out my website: http://gilly.space\n")
    print("Delay: {} Seconds".format(params.background_update_delay_seconds))
    # print("Coronagraph Mode: {} \n".format(params.mode()))
    
    if params.is_debug():
        print("DEBUG MODE\n")


def __debug_mode(params):
    """Run the program in a way that will break"""
    while True:
        params.__execute_switch()


def __run_mode(params):
    """Run the program in a way that won't break"""
    
    fail_count = 0
    fail_max = 10
    
    while True:
        try:
            __execute_switch(params)
        except (KeyboardInterrupt, SystemExit):
            print("\n\nOk, I'll Stop. Doot!\n")
            break
        except Exception as error:
            fail_count += 1
            if fail_count < fail_max:
                print("I failed, but I'm ignoring it. Count: {}/{}\n\n".format(fail_count, fail_max))
                print(error)
                continue
            else:
                print("Too Many Failures, I Quit!")
                sys.exit(1)


def __execute_switch(params):
    """Select which data source to draw from"""
    theSun = sb.Sunback(params)
    if params.run_type().casefold() == "web".casefold():
        WebExecute(params).execute()
    elif params.run_type().casefold() == "mr".casefold():
        theSun.mr_execute()
    elif params.run_type().casefold() == "jp".casefold():
        theSun.jp_execute()
    elif params.run_type().casefold() == "fido".casefold():
        theSun.fido_execute()
