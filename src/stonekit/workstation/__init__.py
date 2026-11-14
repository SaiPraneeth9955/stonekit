"""StoneKit research workstation: a local, offline web application over the frozen StoneKit pipeline.

Layers: `services` (read-only access to run outputs and study evidence), `jobs` (background analyses through
scripts/demo_case.py), `report` (printable report) and `server` (HTTP API and static front end). Calibrated ranges
come from stonekit.t1.case_ranges, a thin wrapper around the frozen uncertainty engine. It presents the frozen research pipeline; nothing is refitted.
"""
