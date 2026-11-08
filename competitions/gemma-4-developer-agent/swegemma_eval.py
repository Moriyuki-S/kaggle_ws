"""Run `swegemma eval` with the context settings that scoring adds on top of the CLI.

Scoring (scripts/inference.py, HARNESS_README section 7.2) compacts old events and caches
the prompt prefix, so that long sessions stay inside the 32,768-token window. The CLI leaves
both unset, which makes long sessions fail with ContextWindowExceededError instead.
Takes the same arguments as `swegemma eval`.
"""

import sys

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import EventsCompactionConfig
from swegemma import cli
from swegemma.config import EvalConfig


def scoring_config(*args, **kwargs) -> EvalConfig:
    kwargs.setdefault(
        "events_compaction_config",
        EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14_336, event_retention_size=5
        ),
    )
    kwargs.setdefault(
        "context_cache_config",
        ContextCacheConfig(min_tokens=2_048, ttl_seconds=1_800, cache_intervals=10),
    )
    return EvalConfig(*args, **kwargs)


if __name__ == "__main__":
    cli.EvalConfig = scoring_config
    sys.exit(cli.main(["eval", *sys.argv[1:]]))
