# SPDX-License-Identifier: Apache-2.0
"""GDN async opt-ins shared by the scheduler, factory and worker engine.

Explicit extra_config wins, including False. The environment fallback keeps
the pinned prototype launcher's opt-ins usable during upstream migration.
"""

# Standard
import os


def _enabled(config, name, legacy_name):
    value = config.get_extra_config_value(name, None)
    if value is None:
        return os.environ.get(legacy_name, "0") == "1"
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def gdn_async_store_enabled(config):
    return _enabled(config, "gdn_async_store", "LMCACHE_GDN_ASYNC_EXPERIMENTAL")


def gdn_async_restore_enabled(config):
    return _enabled(config, "gdn_async_restore", "LMCACHE_GDN_FULL_ASYNC_EXPERIMENTAL")


def gdn_async_staging_mib(config):
    value = config.get_extra_config_value("gdn_async_staging_mib", None)
    if value is None:
        value = os.environ.get("LMCACHE_GDN_ASYNC_STAGING_MIB", "256")
    if type(value) not in (int, str):
        raise ValueError("GDN staging budget must be an integer")
    mib = int(value)
    if not 64 <= mib <= 1024:
        raise ValueError("GDN staging budget must be 64..1024 MiB per rank")
    return mib
