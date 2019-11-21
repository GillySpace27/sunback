from mpi4py import MPI
from numpy import arange
from numpy import zeros_like
import numpy as np
defResponse = None

def poolMPI(work_list, do_work, useBar = False, order = False):
    """Execute the work in parallel and return the result in a list
    work_list: (list) Parameters to be run
    do_work: (function) to be passed the items in work_list
    useBar: (Bool) Display a progress bar (requires progressBar.py)
    order: (Bool) make function return two outputs: list of results; list of job ID for descrambling.
    
    returns: unordered list of results returned by do_work
    """
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    size = MPI.COMM_WORLD.Get_size() 

    if rank == 0:
        all_dat, indices = master(work_list, useBar)
        if order: return reorderScalar(all_dat, indices)
        return all_dat
    else:
        slave(do_work)
        return None

def reorderScalar(array, indices):
    """Unscramble the output, given a scalar array or list of scalars"""  
    sorted = np.argsort(indices)
    newArray = [array[ii] for ii in sorted]
    newInds = [indices[ii] for ii in sorted]
    assert cs(newInds)
    return newArray
    # inds = np.asarray(indices, dtype=np.int)
    # import pdb; pdb.set_trace()
    # array[inds] = array
    # return array

def cs(array):
	trutharray = np.asarray(array[1:])-np.asarray(array[:-1]) == 1
	return not False in trutharray
	
def set_response(func):
    """Set what will be done by the master after each data is received.
       Takes as input a function of one variable: the result of a single core's do_work
    """
    global defResponse
    defResponse = func

#####################
    
def master(wi, useBar):
    """Master process primary loop"""
    WORKTAG = 0
    DIETAG = 1
    all_data = []
    indices = []
    size = MPI.COMM_WORLD.Get_size()
    current_work = __Work__(wi) 
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    
    bar = None
    if useBar:
        try: 
            import progressBar as pb
            bar = pb.ProgressBar(len(wi))
            bar.display()
        except: 
            print("Progress Bar file not found")
            bar = None
    
    
    for i in range(1, size): 
        anext = current_work.get_next_item() 
        if anext is None: break
        comm.send(anext, dest=i, tag=WORKTAG)

    while 1:
        anext = current_work.get_next_item()
        if anext is None: break
        data = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        postReceive(data[0], bar)
        all_data.append(data[0])
        indices.append(data[1])
        comm.send(anext, dest=status.Get_source(), tag=WORKTAG)


    for i in range(1,size):
        data = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        postReceive(data[0], bar)
        all_data.append(data[0])
        indices.append(data[1])

    for i in range(1,size):
        comm.send(None, dest=i, tag=DIETAG)
        
    if bar is not None: bar.display(True)
    return all_data, indices
    
    
def slave(do_work):
    """Slave process primary loop"""
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    while 1:
        data = comm.recv(None, source=0, tag=MPI.ANY_TAG, status=status)
        # print(data)
        if status.Get_tag(): break
        comm.send([do_work(data[0]), data[1]], dest=0)
    
class __Work__():
    """Generator for jobs"""
    def __init__(self, work_items):
        self.work_items = work_items[:] 
        self.currInd = 0

    def get_next_item(self):
        if len(self.work_items) == 0:
            return None
        out = [self.work_items.pop(0), self.currInd]
        self.currInd += 1
        return out

def postReceive(data, bar):
    """Have the master do something in response to receiving a completed job from a slave"""
    if defResponse is not None: defResponse(data)
    if bar is not None: 
        bar.increment()
        bar.display()
        
#####################

def testRun(N = 10):
    """Create N tasks for the MPI pool and do testwork on them"""
    work = arange(N).tolist()
    all_dat = poolMPI(work, testwork)
    print(all_dat)
 
def testwork(num):
    """Find the cube of a number"""
    return num**3

