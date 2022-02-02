"""
A package for combining line transitions based on the natural breaking points in their frequency distribution
and a desired threshold.
"""
from .freqcomb import combine_transitions, get_mini_df, query_splat, plot_freq_dist
__all__ = ["combine_transitions", "get_mini_df", "query_splat", "plot_freq_dist"]