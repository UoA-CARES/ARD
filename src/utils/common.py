"""
Common utilities for the ARD pipeline.

The old SSH machine-pool helpers (``load_machine_pool`` / ``test_machine_pool_ssh``)
were removed: ARD no longer SSHes into machines itself. Worker registration and
dispatch are owned by the Parallel Coordination System coordinator, which ARD
talks to via :class:`src.evaluation.CoordinatorClient`.
"""
