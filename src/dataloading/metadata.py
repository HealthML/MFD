import numpy as np
import pandas as pd


def make_one_hot(col):
    col = col.factorize()[0]
    one_hot = np.zeros((len(col), col.max() + 1))

    for i, val in enumerate(col):
        one_hot[i, val] = 1

    return one_hot.tolist()


def make_multi_hot(indices, total):
    one_hot = np.zeros(total)
    for indice in indices:
        one_hot[indice] = 1
    return one_hot


def categorize_multilabel_column(col, return_labels = False, labels = None):
    
    all_vals = labels
    if all_vals is None:
        # get all possible single values
        all_vals = set(sum([str(val).split(',') for val in col.unique()], []))
        # make it a list so we can index it
        all_vals = sorted(list(all_vals))
    else:
        observed_vals = set(sum([str(val).split(',') for val in col.unique()], []))
        unobserved = []
        for o in observed_vals:
            if o not in all_vals:
                unobserved.append(o)
        assert len(unobserved) == 0, f'The data contain unexpected values: {unobserved}'
            

    onehot_vals = col.apply(
            lambda val: make_multi_hot(
                [all_vals.index(c) for c in str(val).split(',')],
                len(all_vals),
            )
        )
    
    if return_labels:
        return onehot_vals, all_vals
    else:
        return onehot_vals
        


class MetadataLoader:
    def __init__(
            self,
            metadata_file,
            label_ids=None,
            **kwargs,
    ):
        md = pd.read_csv(metadata_file, sep='\t')
        md.rename(columns={c: c.replace('.', '_') for c in md.columns}, inplace=True)
        md.set_index('File_accession', drop=False, inplace=True)

        # necessary preprocessing
        def get_median_age_and_unit(row):
            if row['proc_Biosample_life_stage'] == 'unknown':
                return -1, 'unknown'

            df = md.loc[
                (md['Biosample_organism'] == row['Biosample_organism'])
                & (md['proc_Biosample_life_stage'] == row['proc_Biosample_life_stage'])
                & (md['proc_age_bin'] != 'unknown')
                ].copy()
            
            df['proc_age_bin'] = df['proc_age_bin'].astype(float)

            return df['proc_age_bin'].median(), df['proc_age_bin_units'].mode()[0]

        md['proc_age_bin'] = md['proc_age_bin'].map(lambda x: '90' if x == '90 or above' else x)

        rows_unkn_age = md['proc_age_bin_units'].isna()
        md.loc[rows_unkn_age, ['proc_age_bin', 'proc_age_bin_units']] = md.loc[rows_unkn_age].apply(
            get_median_age_and_unit,
            axis=1,
            result_type="expand",
        )
        md['proc_age_bin'].fillna(-1, inplace=True)

        if label_ids is not None:
            md = md.loc[label_ids]
        self.df_raw = md.copy()

        # TODO: refactor 
        # Code below is obsolete!!
        # data coding is handled by the models.MetadataEmbedding class

        # technical factors
        md['Experiment_date_released'] = make_one_hot(pd.DatetimeIndex(md['Experiment_date_released']).year)
        md['proc_Assay_lvl1'] = make_one_hot(md['proc_Assay_lvl1'])
        md['Library_lab'] = make_one_hot(md['Library_lab'])

        # biological factors
        md['Biosample_term_name'] = make_one_hot(md['Biosample_term_name'])
        md['Biosample_organism'] = make_one_hot(md['Biosample_organism'])
        md['proc_target'] = make_one_hot(md['proc_target'])
        md['proc_Biosample_life_stage'] = make_one_hot(md['proc_Biosample_life_stage'])

        md['Biosample_organ_slims'] = categorize_multilabel_column(md['Biosample_organ_slims'])
        md['Biosample_system_slims'] = categorize_multilabel_column(md['Biosample_system_slims'])
        md['Biosample_developmental_slims'] = categorize_multilabel_column(md['Biosample_developmental_slims'])

        # handle different age units by just concatenating the age bin with the unit, e.g.:
        # 13.5day, 60year
        # and categorizing that - cleaner ideas are welcome
        md['proc_age_bin_w_units'] = md['proc_age_bin'] + md['proc_age_bin_units']
        md['proc_age_bin_w_units'] = make_one_hot(md['proc_age_bin_w_units'])

        self.df = md
        self.cols_processed = [
            'Biosample_organ_slims',
            'Biosample_system_slims',
            'Biosample_developmental_slims',
            'Biosample_term_name',
            'Biosample_organism',
            'proc_target',
            'proc_Biosample_life_stage',
            'proc_age_bin_w_units',
            'Experiment_date_released',
            'proc_Assay_lvl1',
            'Library_lab',
        ]
        self.cols_biological = self.cols_processed[:-3]
        self.cols_technical = self.cols_processed[-3:]

    def _get_columns(self, cols):
        if cols == 'all':
            columns = self.cols_processed
        elif cols == 'biological':
            columns = self.cols_biological
        elif cols == 'technical':
            columns = self.cols_technical
        else:
            raise ValueError(f'Unknown columns type: {cols}')
        return columns

    def experiment_index_to_metadata(self, exp_idx, cols='all'):
        """
        Returns a DataFrame row corresponding to the experiment index in the DataFrame.
        exp_idx is an integer, NOT the experiment name!
        """
        columns = self._get_columns(cols)
        return self.df.iloc[exp_idx][columns]

    def experiment_index_to_metadata_vector(self, exp_idx, cols='all'):
        metadata_row = self.experiment_index_to_metadata(exp_idx=exp_idx, cols=cols)
        metadata_vector = metadata_row.values
        return metadata_vector

    def get_metadata(self, cols='all'):
        columns = self._get_columns(cols)
        return self.df[columns]

    def get_metadata_vectors(self, cols='all'):
        metadata_df = self.get_metadata(cols=cols)
        return [row.values for _, row in metadata_df.iterrows()]

    def get_metadata_vectors_biological(self):
        return self.get_metadata_vectors(cols='biological')

    def get_metadata_vectors_technical(self):
        return self.get_metadata_vectors(cols='technical')

    def get_metadata_biological(self):
        return self.get_metadata(cols='biological')

    def get_metadata_technical(self):
        return self.get_metadata(cols='technical')

    @property
    def size_biological(self):
        return self.experiment_index_to_metadata_vector(0, cols='biological').shape[0]

    @property
    def size_technical(self):
        return self.experiment_index_to_metadata_vector(0, cols='technical').shape[0]

    @property
    def size_all(self):
        return self.size_biological + self.size_technical


if __name__ == '__main__':
    md_loader = MetadataLoader(metadata_file='data/processed/221111_encode_metadata_processed.tsv')

    print(md_loader.cols_biological)
    print(md_loader.cols_technical)
    print(md_loader.get_metadata_biological())
    print(md_loader.get_metadata_technical())

    print(md_loader.get_metadata_vectors())
    print(md_loader.get_metadata_vectors_biological())
    print(md_loader.get_metadata_vectors_technical())

    print(md_loader.size_biological)
    print(md_loader.size_technical)
    print(md_loader.size_all)

    print(md_loader.experiment_index_to_metadata(2))
    print(md_loader.experiment_index_to_metadata(111))
