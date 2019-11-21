import sys
from datetime import datetime
import struct

class ProgressBar():

    """This class can be used to display a text-based progress bar.
    
    The class constructor is given some number to count up to. The
    update(current) method is used to tell the progress bar what
    number has been counted to so far, and the display() method
    prints the current progress to the screen.
    
    The displayed progress bar includes elapsed time, percentage completion,
    the rate at which the counting variable is increasing, the bar
    itself, and the ETA for completion.
    
    ETA and rate are based on speed over the last 10 seconds, and are
    updated only once per second to minimize jitter.
    """
    elapsedString = "{0}:{1:02}:{2:02}"  # e.g. "1:23:45" or "123:45:67"
    etaString = " ETA {0}:{1:02}:{2:02}"  # e.g. " ETA 1:23:45"
                                          # or " ETA 123:45:67"
    noEtaString = " ETA -:--:--"  # For when the ETA is still settling
                                  # or is infinite
    percentString = " {0:3}%"  # e.g. " 100%" or "   1%"
    rateString = " [{0:3}/m]"  # e.g. " [  5/s]" or " [1234/s]"
    noRateString = " [---/s]"  # For when the rate is settling
    
    disable = False
    
    def __init__(self, target, updateRate=0.2, color = None):

        
        if color is not None: 
            try:
                import colorama
                colorama.init()

                if color.casefold() in 'red': self.cs = "\033[91m"
                if color.casefold() in 'green': self.cs = "\033[92m"
                if color.casefold() in 'blue': self.cs = "\033[94m"
                if color.casefold() in 'cyan': self.cs = "\033[96m"
                if color.casefold() in 'purple': self.cs ="\033[95m"
                if color.casefold() in 'yellow': self.cs = "\033[93m"

                self.ce = '\033[00m'
            except: self.cs = self.ce = ''
        else: self.cs = self.ce = ''

        self.target = float(target)
        self.updateRate = updateRate
        self.current = 0
        self.lastEta = self.noEtaString
        self.lastRate = self.noRateString
        self.benchmarks = [(datetime.now(), 0)] * 20
        self.start = None
        self.lastUpdate = None
        self.lastDisplay = None
        self.minAdjust = 60
    
    def setTarget(self, target):
        self.target = target
    
    def update(self, current):
        if(self.start is None):
            self.start = self.lastUpdate = datetime.now()
        self.current = float(current)

    def increment(self, change=1):
        self.update(self.current + change)
    
    def display(self, force=False):

        if self.disable:
            return
        now = datetime.now()
        current = self.current
        if(self.start is None):
            self.start = self.lastUpdate = now
        
        if self.lastDisplay is not None and not force and \
                self.totalSeconds(now - self.lastDisplay) < self.updateRate:
            return
        self.lastDisplay = now
        # Prepare the percentage completion display
        try:
            percent = current / self.target * 100
        except ZeroDivisionError:
            percent = 100.
        percentString = self.percentString.format(int(percent))
        
        # Prepare the elapsed time display
        elapsedSeconds = self.totalSeconds(now - self.start)
        timeH, timeM, timeS = self.formatTime(int(elapsedSeconds))
        elapsedString = self.elapsedString.format(timeH, timeM, timeS)
        
        etaString = self.lastEta
        rateString = self.lastRate
     
        # If it's been more than a second since the last progress benchmark
        # was stored, let's update it as well as the ETA and progress rate.
        if(self.totalSeconds(now - self.lastUpdate) > 2):
            self.lastUpdate = now
            self.benchmarks.insert(len(self.benchmarks), (now, current))
            benchmark = self.benchmarks.pop(0)
            secondsSinceBenchmark = self.totalSeconds(now - benchmark[0])
            benchmarkValue = benchmark[1]
            
            incrementRate = ((current - benchmarkValue) / secondsSinceBenchmark)

            if force:
                incrementRate = current / (self.totalSeconds(now - self.start))
            # Prepare the eta display
            if(elapsedSeconds < 1 or percent == 0 or incrementRate == 0):
                # For the first little while, the ETA is absolute nonsense.
                etaString = self.noEtaString
            else:
                eta = int((self.target - current) / incrementRate)
                etaH, etaM, etaS = self.formatTime(eta)
                etaString = self.etaString.format(etaH, etaM, etaS)
            self.lastEta = etaString
        
            # Prepare the rate display
            if(elapsedSeconds < 1 or percent == 0):
                rateString = self.noRateString
            elif incrementRate < 1:
                rateString = self.rateString.format(round(incrementRate * self.minAdjust, 1))
            else:
                rateString = self.rateString.format(int(incrementRate * self.minAdjust))
            self.lastRate = rateString
        

        iterString = ' [' + str(int(self.current)) + '/' + str(int(self.target)) + ']'

        # Prepare the progress bar display
        terminalSize = getTerminalSize()[0]
        #print(terminalSize)
        #import time
        #print("Sleepytime")
        #sys.stdout.flush()
        #time.sleep(20)   
        # The 4 accounts for an initial space, two brackets, and a
        # 'rest space' at the end of the line to keep the cursor from
        # filling the line and moving the cursor to the next line.
        availableSize = terminalSize - 4 - len(elapsedString) \
                        - len(percentString) - len(etaString) - len(rateString) - len(iterString)
        if percent > 100:
            percent = 100
        nTickMarks = int(availableSize * 2 * percent / 100)
        progressString = ' ['
        progressString += '=' * int(nTickMarks / 2)
        progressString += '-' * (nTickMarks % 2)
        #progressString += ' ' * int(availableSize - (nTickMarks + 1) / 2)
        progressString += ' ' * int(availableSize - len(progressString) + 2)
        progressString += ']'


        
        # An \r character moves the cursor to the beginning of the line
        # so we can re-print the line.
        print("\r", end=self.cs)
        print(elapsedString + percentString + iterString + rateString \
              + progressString + etaString, end=self.ce)
        # Without this, Python may buffer the output but not actually show it,
        # since we haven't finished a line.
        if force: print('')
        sys.stdout.flush()
        return
    
    def formatTime(self, elapsedSeconds):
        """Converts a number of seconds to hours, minutes, and seconds."""
        hours = elapsedSeconds // 3600
        minutes = (elapsedSeconds % 3600) // 60
        seconds = (elapsedSeconds % 3600) % 60
        return hours, minutes, seconds
    
    def totalSeconds(self, t):
        """Converts a timeDelta to a number of seconds. timeDelta instances
        only have this built-in for python > 2.7"""
        return ((t.microseconds + (t.seconds + t.days * 24 * 3600) * 10 ** 6)
                / 10. ** 6)
 
"""
Terminal size code from http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
"""

import platform
import subprocess

def getTerminalSize():
    try:
        current_os = platform.system()
    except:
        # Ugly workaround: platform.system() raises an exception
        # (related to SIGCHLD handling, I think) on Dahl, so we
        # can just assume we're on Dahl--ergo, Linux--if this
        # raises an exception.
        current_os = 'Linux'
    tuple_xy = None
    if current_os == 'Windows':
        tuple_xy = _getTerminalSize_windows()
        if tuple_xy is None:
            pass
            #tuple_xy = _getTerminalSize_tput() ##################THIS WAS THROWING A WARNING I DONT WANT
            # needed for window's python in cygwin's xterm!
    if current_os == 'Linux' or current_os == 'Darwin' \
       or current_os.startswith('CYGWIN'):
        tuple_xy = _getTerminalSize_linux()
    if tuple_xy is None:
        tuple_xy = (80, 25)      # default value
    return tuple_xy

def _getTerminalSize_windows():
    res = None
    try:
        from ctypes import windll, create_string_buffer

        # stdin handle is -10
        # stdout handle is -11
        # stderr handle is -12

        h = windll.kernel32.GetStdHandle(-12)
        csbi = create_string_buffer(22)
        res = windll.kernel32.GetConsoleScreenBufferInfo(h, csbi)
    except:
        return None
    if res:
        (bufx, bufy, curx, cury, wattr,
         left, top, right, bottom, maxx, maxy) = struct.unpack("hhhhHhhhhhh",
                                                               csbi.raw)
        sizex = right - left + 1
        sizey = bottom - top + 1
        return sizex, sizey
    else:
        return None

def _getTerminalSize_tput():
    # get terminal width
    # src:
    # http://stackoverflow.com/questions/263890/how-do-i-find-the-width-height-of-a-terminal-window
    try:
        proc = subprocess.Popen(
            ["tput", "cols"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        output = proc.communicate(input=None)
        cols = int(output[0])
        proc = subprocess.Popen(
            ["tput", "lines"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        output = proc.communicate(input=None)
        rows = int(output[0])
        return (cols, rows)
    except:
        return None

def _getTerminalSize_linux():
    import fcntl, termios, struct, os
    def ioctl_GWINSZ(fd):
        try:
            cr = struct.unpack('hh',
                               fcntl.ioctl(fd, termios.TIOCGWINSZ, '1234'))
        except:
            return None
        return cr
    cr = None
    
    try:
        cr = (os.environ['LINES'], os.environ['COLUMNS'])
    except KeyError:
        pass
    
    if not cr:
        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
        except:
            return None
        finally:
            try:
                os.close(fd)
            except:
                return None
    
    return int(cr[1]), int(cr[0])

if __name__ == "__main__":
    from time import sleep
    pb = ProgressBar(10005)
    pb.display()
    for i in range(0, 10006):
        sleep(0.003)
        pb.update(i)
        pb.display()
