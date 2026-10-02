"""pytest configuration for sunback/__tests__ (SB-2).

These three files stay on disk but are not collected:
- test_sunback.py imports ``movie.dep.sunback``, a module that no longer exists,
  and reaches the network.
- test_parameters.py imports a top-level ``parameters`` module that no longer exists.
- "test registry.py" is a Windows winreg script, not a test.
"""
collect_ignore = ["test_sunback.py", "test_parameters.py", "test registry.py"]
