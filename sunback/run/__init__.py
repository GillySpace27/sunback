"""
Empty init file in case you choose a package besides PyTest such as Nose which may look for such a file
"""
from sunback.run.run import Runner
from sunback.run.run import SingleRunner
# from src.run.run_panhelio import run_QRN
__all__ = [Runner, SingleRunner]


# #  TODO: Make this not bad
# def runner(p):
#     return None
