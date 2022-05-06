import freqcomb as fc
import pandas as pd
import os

mol_dict = {'CN': ' CN v = 0 ',
            'C15N': ' C15N ',
            '13CN': ' 13CN ',
            'OH': ' OH v=0 ',
            'CO': ' CO v = 0 ',
            '13CO': ' 13CO v = 0 ',
            'C18O': ' C18O ',
            'C17O': ' C17O ',
            '13C18O': ' 13C18O ',
            '13C17O': ' 13C17O ',
            'CS': ' CS v = 0 ',
            'C34S': ' C34S v = 0 ',
            '13CS': ' 13CS v = 0 ',
            'SO': ' SO 3Σ v = 0 ',
            'H2O': ' H2O v=0 ',
            '18OO': ' 18OO ',
            'HDO': ' HDO ',
            'H218O': ' H218O v=0 ',
            'HCO+': ' HCO\+ v=0 ',
            'DCO+': ' DCO\+ v = 0 ',
            'H13CO+': ' H13CO\+ ',
            'HC18O+': ' HC18O+',
            'HCN': ' HCN v=0 ',
            'DCN': ' DCN v = 0 ',
            'H13CN': ' H13CN v = 0 ',
            'HC15N': ' HC15N v = 0 ',
            'HNC': ' HNC v=0 ',
            'DNC': ' DNC ',
            'H2S': ' H2S ',
            'N2H+': ' N2H\+ v = 0 ',
            'N2D+': ' N2D\+ ',
            'CCH': ' CCH v = 0 ',
            'CCD': ' CCD ',
            'NH3': ' NH3 v=0 ',
            'H2CO': ' H2CO ',
            'H2CS': ' H2CS ',
            'HC3N': ' HC3N v=0 ',
            't-HCOOH': ' t-HCOOH ',
            'c-C3H2': ' c-HCCCH  v=0 ',
            'CH3OH': ' CH3OH vt = 0 ',
            'CH3CN': ' CH3CN v = 0 ',
            'CH3OCH3': ' CH3OCH3 ',
            'CH2CN': ' CH2CN '}

df_list = []
for mol_simple, mol_splat in mol_dict.items():
    print(mol_simple)

    # Query Splatalogue for a given molecule, with E_up < 500 K and CDMS/JPL intensity > 10^-6
    mytable = fc.query_splat(mol_splat, energy_max=500.0, intensity_lower_limit=-6)

    # Convert the Astropy table to a minimal format as a pandas.DataFrame
    mol_df = fc.get_mini_df(mytable)

    # Show and save plots of frequency distribution in subfolder './plots'
    fc.plot_freq_dist(frequencies=mol_df['Freq'],
                      bin_width=10.0,
                      title=mol_simple,
                      show_fig=False,
                      save_fig='{}_freq_dist'.format(mol_simple))
    # Combine transitions based on the natural breaking points in their frequency distribution
    # (using a kernel density estimator) and a desired threshold in km/s units. The results can
    # be saved as a csv table by setting the 'save_combined' parameter to True. The transition
    # groupings can optionally be saved in a csv file by setting save_grouped=True.
    combined_df = fc.combine_transitions(mol_df,
                                         vel_threshold=10.0,
                                         save_combined=True,
                                         save_grouped=True,
                                         filename_prefix=mol_simple)
    df_list.append(combined_df)

# Combine all DataFrames into one
all_mol_df = pd.concat(df_list)
save_path = './tables'
all_mol_df.to_csv(os.path.join(save_path, "allmols_combined_transitions.csv"), index=False)