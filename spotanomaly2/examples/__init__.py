# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Self-contained, publicly runnable example integrations.

Code here demonstrates how to drive the pipeline against real, public data
sources without the proprietary credentials the production `domain` services
require. It is deliberately kept out of `domain/` because it is illustrative
glue, not core anomaly-detection logic — but it lives inside the installed
package so config files can reference it by dotted path (e.g. the pluggable
``primary.fetcher`` in ``config/entsoe_example.yaml``).
"""
