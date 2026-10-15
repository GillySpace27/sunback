"""Slim wallpaper client and dome tool (SB-21): standard library only.

Nothing in this package imports numpy, requests, boto3 or the science stack, so
``python -m sunback.client`` starts in well under a second. ``sunback-run`` still
points at ``sunback.run.run_client_background.run_client`` until Gilly has
compared the two side by side on a Mac.
"""
