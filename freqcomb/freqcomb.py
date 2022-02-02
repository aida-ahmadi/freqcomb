"""
Combine line frequencies based on their natural breaking points and a desired threshold.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy import units as u
from astropy import constants as const
from astroquery.splatalogue import Splatalogue, utils
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


def query_splat(mol, min_freq=84.0*u.GHz, max_freq=950.0*u.GHz, energy_max=None, intensity_lower_limit=None):
    """
    Uses astroquery's Splatalogue Class (query_lines function) to query lines in the ALMA range of 84 - 950 GHz.
    The JPL catalogue is queried first and if it returns no results, the CDMS database is queried.

    Parameters
    ----------
    mol : str
         Name of the chemical to search for. The nomenclature follows astroquery's 'query_lines' function.
    min_freq : astropy.units, optional
         (Default value = 84.0 GHz)
         Minimum frequency
    max_freq : astropy.units, optional
         (Default value = 950.0 GHz)
         Maximum frequency
    energy_max : float64, optional
         Include transitions with energies up to E_upper in Kelvin units
    intensity_lower_limit : float, optional
         Include transitions with intensities higher than this limit given in base 10 logarithm of the integrated
         intensity ('cdms_jpl' type).

    Returns
    -------
    astropy.table containing the query results

    """
    mytable = Splatalogue.query_lines(min_frequency=min_freq,
                                      max_frequency=max_freq,
                                      energy_max=energy_max,
                                      energy_type='eu_k',
                                      chemical_name=mol,
                                      energy_levels=['el4'],
                                      line_strengths=['ls1'],
                                      line_lists=['JPL'],
                                      intensity_lower_limit=intensity_lower_limit,
                                      intensity_type='cdms_jpl')
    if not mytable:
        mytable = Splatalogue.query_lines(min_frequency=min_freq,
                                          max_frequency=max_freq,
                                          energy_max=energy_max,
                                          energy_type='eu_k',
                                          chemical_name=mol,
                                          energy_levels=['el4'],
                                          line_strengths=['ls1'],
                                          line_lists=['CDMS'],
                                          intensity_lower_limit=intensity_lower_limit,
                                          intensity_type='cdms_jpl')
    return mytable


def get_mini_df(splat_table):
    """
    Concatenate astroquery's Splatalogue queried table into their minimalist form, add the 'CDMS/JPL Intensity' column,
    and convert the table into a pandas.DataFrame

    Parameters
    ----------
    splat_table : astropy.table
         The Splatalogue table

    Returns
    -------
    pandas.DataFrame

    """
    if not splat_table:
        print('The provided Astropy table is empty.')
    mini_splat_table = utils.minimize_table(splat_table)
    # Add the 'CDMS/JPL Intensity' column:
    if 'CDMS/JPL Intensity' in splat_table.colnames:
        mini_splat_table.add_column(splat_table['CDMS/JPL Intensity'])
    mini_splat_df = mini_splat_table.to_pandas()
    return mini_splat_df


def plot_freq_dist(frequencies, bin_width=10.0, title=None, show_fig=True, save_fig=None, save_path='./plots'):
    """
    Plots a histogram of the frequency distribution, showing the exact position of un-binned data points (rugplot)

    Parameters
    ----------
    frequencies : list of numbers, astropy.table.Column, or pandas.Series (i.e. column of a pandas.DataFrame)
         Frequencies to plot (in GHz).
    bin_width : float64, optional
         (Default value = 10.0)
         Histogram bin width, in the same units as the provided frequencies.
    title : str, optional
         Name used for the title of the plot.
    show_fig : bool, optional
         (Default value = True)
         Display the plot (showfig=True) or not (showfig=False).
    save_fig : str, optional
         Filename (without an extension) for the plot to be saved as. Default file extension is PDF.
    save_path : str, optional
         The directory where the figure should be saved. By default, the figure is saved in a subdirectory called
         'plots' within the current working directory. Default quality is dpi=300.

    """
    if frequencies is None or len(frequencies) == 0:
        print('Nothing to plot.')
        return
    else:
        bins = np.arange(min(frequencies), 1.1*max(frequencies), bin_width)
        sns.displot(frequencies, bins=bins, rug=True, rug_kws={"color": "r", "alpha": 0.5})
        plt.axhline(1.0, color='black', ls='--')
        if title is not None:
            plt.title(title)
        plt.xlabel('Frequency (GHz)')
        suffix = '.pdf'
        if save_fig is not None:
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, save_fig + suffix), dpi=300, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close()


def do_KDE(frequencies):
    """
    Run scikit-learn's kernel density estimator for the distribution of frequencies using a Gaussian
    kernel with a bandwidth of 0.1

    Parameters
    ----------
    frequencies : numpy.ndarray
         Frequencies reshaped into a column for the KDE algorithm

    Returns
    -------
    freq_range : numpy.ndarray
         Array of frequencies arranged from minimum to maximum values in steps of 0.1
    likelihoods : numpy.ndarray
         Log-likelihood of each sample in freq_range

    """
    if len(frequencies) > 1:
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(frequencies)
        freq_range = np.arange(min(frequencies), max(frequencies), 0.1)
        likelihoods = kde.score_samples(freq_range.reshape(-1, 1))
    else:
        freq_range = None
        likelihoods = None
    return freq_range, likelihoods


def split_arrays(input_array, indices):
    """
    Split Numpy arrays at specific indices

    Parameters
    ----------
    input_array : numpy.ndarray
         The array to split
    indices : list
         Indices at which to split the numpy arrays

    Returns
    -------
    numpy.ndarray

    """
    split_elements = []
    for e, element in enumerate(input_array):
        if indices[e] is not None:
            split_elements.extend(np.split(element, indices[e]))
        else:
            split_elements.append(element)
    return split_elements


def get_indices(freq_array, vel_threshold):
    """
    Get the indices at which to split arrays further

    Parameters
    ----------
    freq_array : numpy.ndarray
         Array of frequencies
    vel_threshold : float
         If adjacent elements in a given grouped frequencies are separated by more than this value in
         velocity (in km/s) the arrays will need further splitting than the groupings already done by the KDE.
         A value of 0.0 will avoid further splitting.

    Returns
    -------
    list of indices

    """
    indices_list = []
    for freqs in freq_array:
        frequency_diff = np.diff(freqs)
        if len(frequency_diff) > 0:
            delta_v = (const.c * frequency_diff / np.delete(freqs, 0)).to(u.km/u.s)
            indices = list(np.where(delta_v.to(u.km/u.s).value > vel_threshold)[0] + 1)
            indices_list.extend([indices])
        else:
            indices_list.append(None)
    return indices_list


def group_arrays(input_array, frequencies):
    """
    Run scikit-learn's kernel density estimator for the distribution of frequencies and group arrays
    based on the likelihood minima/maxima.

    Parameters
    ----------
    input_array : numpy.ndarray
         The array to group
    frequencies : numpy.ndarray
         Frequencies reshaped into a column for the KDE algorithm

    Returns
    -------
    numpy.ndarray that has been grouped based on the distribution of frequencies

    """
    grouped_array = []

    # Do the Kernel Density estimation and get the frequency minima/maxima for the distribution
    freq_range, likelihoods = do_KDE(frequencies)

    if freq_range is not None and likelihoods is not None:
        mi, ma = argrelextrema(likelihoods, np.less)[0], argrelextrema(likelihoods, np.greater)[0]

        # Group data based on the minima/maxima
        if mi.any():
            # The first elements are where frequencies are < the first minimum:
            grouped_array.append(input_array[frequencies < freq_range[mi][0]])

            # The rest of the elements are where frequencies are between two consecutive minima:
            if len(mi) > 0:
                for i in range(len(mi) - 1):
                    grouped_array.append(input_array[(frequencies >= freq_range[mi][i]) *
                                                     (frequencies < freq_range[mi][i + 1])])

                # The last elements are where frequencies are > the last minimum
                grouped_array.append(input_array[(frequencies >= freq_range[mi][len(mi) - 1])])
    else:
        grouped_array = input_array
    return grouped_array


def combine_transitions(input_df, vel_threshold=10.0, save_grouped=False, save_combined=True, filename_prefix=None,
                        save_path='./tables'):
    """
    Given a pandas.DataFrame with a 'Freq' column, group transitions based on their natural breaking
    points in their frequency distribution using a Kernel Density Estimator, split transitions further if
    needed based on a velocity threshold between neighbouring transitions, and combine each group of transitions
    into one single transition (one with the lowest E_up ('EU_K'), or highest intensities ('CDMS/JPL Intensity')
    if those columns exist).

    Parameters
    ----------
    input_df : pandas.DataFrame
         DataFrame containing line frequencies and other parameters

         Index:
            RangeIndex
         Columns:
            Name: Freq, dtype: float64, description: Frequency in GHz
            Name: EU_K (optional), dtype: float64, description: Upper energy level in Kelvin
            Name: 'CDMS/JPL Intensity' (optional), dtype: float64, description: base 10 logarithm of intensities
            in CDMS/JPL format
    vel_threshold : float, optional
         (Default value = 10.0 km/s)
         If adjacent elements in a given grouped frequencies are separated by more than this value in
         velocity (in km/s) the arrays will be further splitting than the groupings already done by the KDE.
         A value of 0.0 will avoid further splitting.
    save_grouped : bool, optional
         (Default value = False)
         Save the grouped transitions as a csv table (save_grouped=True) or not (save_grouped=False)
    save_combined : bool, optional
         (Default value = True)
         Save the combined transitions as a csv table (save_combined=True) or not (save_combined=False)
    filename_prefix : str, optional
         Filename prefix (without an extension) for the table to be saved as. Default file extension
         is csv.
    save_path : str, optional
         The directory where the tables should be saved. By default, tables are saved in a subdirectory called
         'tables' within the current working directory.

    Returns
    -------
    pandas.DataFrame of the combined transitions

    """
    if input_df is None or input_df.empty:
        print("The provided DataFrame is empty.")
        return
    else:
        # Check if Freq column exists
        if 'Freq' not in input_df.columns:
            print("Error: The provided DataFrame must contain a column named 'Freq' but does not.")
            return

        # Get frequencies
        frequencies = np.array(input_df['Freq']).reshape(-1, 1)
        grouped_frequencies = group_arrays(frequencies, frequencies)
        indices = get_indices(grouped_frequencies, vel_threshold=vel_threshold)
        split_frequencies = split_arrays(grouped_frequencies, indices)

        # Get energies if they exist
        if 'EU_K' in input_df.columns:
            EU_K = np.array(input_df['EU_K']).reshape(-1, 1)
            grouped_EU_K = group_arrays(EU_K, frequencies)
            split_EU_K = split_arrays(grouped_EU_K, indices)
        else:
            split_EU_K = None
            print("Note: The provided DataFrame does not contain a column named 'EU_K'. "
                  "The frequency combination will be done by keeping the median elements.")

        # Get intensities if they exist
        if 'CDMS/JPL Intensity' in input_df.columns:
            log10_intensities = np.array(input_df['CDMS/JPL Intensity']).reshape(-1, 1)
            grouped_log10_intensities = group_arrays(log10_intensities, frequencies)
            split_log10_intensities = split_arrays(grouped_log10_intensities, indices)
        else:
            split_log10_intensities = None

        grouped_df = pd.DataFrame()
        combined_df = pd.DataFrame()
        for column in input_df:
            # Create numpy arrays for each column
            column_array = np.array(input_df[column]).reshape(-1, 1)

            # Do KDE & group data based on the minima/maxima
            grouped_array = group_arrays(column_array, frequencies)

            # Additional criteria that goes through the groupings that was done by the KDE method, and determines
            # whether further splitting is required. This is done by determining whether two consecutive frequencies
            # within a group are more than 10 km/s apart. If so, the grouping is split even further.
            velsplit_array = split_arrays(grouped_array, indices)

            # Write to a new DataFrame
            grouped_df[column] = velsplit_array

            # Save the DataFrame to CSV file
            if save_grouped:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if filename_prefix is not None:
                    grouped_df.to_csv(os.path.join(save_path, "{}_grouped_transitions.csv".format(filename_prefix)),
                                      index=False)
                else:
                    grouped_df.to_csv(os.path.join(save_path, "grouped_transitions.csv".format(filename_prefix)),
                                      index=False)

            # Reduce the groupings to only one element. If the EU_K column exists and the energies in a given group
            # are not all within 1 Kelvin, then select transition with lowest energy. If the energies are all very
            # similar (to within 1 K), then check whether the CDMS/JPL intensities ('CDMS/JPL Intensity' column) vary
            # significantly. If the intensities are not very similar, then choose the transition with the
            # highest Einstein A coefficient. In the cases where there are no columns corresponding to the energies
            # and CDMS/JPL intensities, or if their values are all very similar, we keep the median elements in a
            # given grouping.
            combined_element = []
            for f, freq in enumerate(split_frequencies):
                if split_EU_K is not None:
                    if any((split_EU_K[f]-np.min(split_EU_K[f])) > 1.0):
                        selected_index = np.argmin(split_EU_K[f])
                    else:
                        selected_index = int(len(velsplit_array[f]) / 2.0)
                        if split_log10_intensities is not None:
                            if any((split_log10_intensities[f]-np.min(split_log10_intensities[f])) > 0.05):
                                selected_index = np.argmax(split_log10_intensities[f])
                else:
                    selected_index = int(len(velsplit_array[f])/2)
                    if split_log10_intensities is not None:
                        if any((split_log10_intensities[f] - np.min(split_log10_intensities[f])) > 0.05):
                            selected_index = np.argmax(split_log10_intensities[f])

                combined_element.append(velsplit_array[f][selected_index])

            # Write to a new DataFrame
            combined_df[column] = combined_element

            # Save the DataFrame to CSV file
            if save_combined:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if filename_prefix is not None:
                    combined_df.to_csv(os.path.join(save_path, "{}_combined_transitions.csv".format(filename_prefix)),
                                       index=False)
                else:
                    combined_df.to_csv(os.path.join(save_path, "combined_transitions.csv".format(filename_prefix)),
                                       index=False)
        return combined_df
