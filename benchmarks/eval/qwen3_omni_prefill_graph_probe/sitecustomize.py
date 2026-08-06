# SPDX-License-Identifier: Apache-2.0
"""Startup hook used only by the graph-probe subprocess."""

from probe import install_from_env


install_from_env()
