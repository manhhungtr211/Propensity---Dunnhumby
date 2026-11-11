"""Immutable result contracts for non-transaction table profiling."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class ProductProfile:
    overview: pl.DataFrame
    cardinality: pl.DataFrame
    frequencies: pl.DataFrame
    hierarchy: pl.DataFrame


@dataclass(frozen=True, slots=True)
class RedemptionProfile:
    overview: pl.DataFrame
    frequencies: pl.DataFrame
    group_distributions: pl.DataFrame
    daily_activity: pl.DataFrame
    return_intervals: pl.DataFrame


@dataclass(frozen=True, slots=True)
class CampaignTableProfile:
    overview: pl.DataFrame
    campaign_counts: pl.DataFrame
    group_distributions: pl.DataFrame


@dataclass(frozen=True, slots=True)
class CausalProfile:
    overview: pl.DataFrame
    code_frequencies: pl.DataFrame
    weekly_activity: pl.DataFrame
    group_distributions: pl.DataFrame


@dataclass(frozen=True, slots=True)
class DemographicProfile:
    overview: pl.DataFrame
    cardinality: pl.DataFrame
    frequencies: pl.DataFrame
    segments: pl.DataFrame


@dataclass(frozen=True, slots=True)
class CouponMappingProfile:
    overview: pl.DataFrame
    group_distributions: pl.DataFrame
    campaign_counts: pl.DataFrame


@dataclass(frozen=True, slots=True)
class CampaignDescriptionProfile:
    overview: pl.DataFrame
    campaigns: pl.DataFrame
    type_counts: pl.DataFrame
    duration_summary: pl.DataFrame

