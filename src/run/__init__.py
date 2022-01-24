"""
Empty init file in case you choose a package besides PyTest such as Nose which may look for such a file
"""
from run.run import Runner
from run.run import SingleRunner
from run.run_panhelio import run_srn
__all__ = [Runner, SingleRunner, run_srn]


# #  TODO: Make this not bad
# def runner(p):
#     return None
