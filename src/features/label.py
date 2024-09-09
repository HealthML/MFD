import h5py
import numpy as np
import gzip

from dataloading.utils import md5


class LabelLoader():
    """
    a label loader should implement the following methods
    """

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        # default
        return self.get_labels(i)

    def get_labels(i):
        '''
        Return labels for region(s) "i" in sparse representation e.g. [1,10,32,100] (or None if no labels are present)
        
        output will have shape (len(i), ?)
        '''
        pass

    def get_label_ids(self):
        '''
        Return the label identifiers 
        '''

    def get_labels_one_hot(i):
        '''
        Return labels for region(s) "i" in 1-hot representation
        
        output will have shape (len(i), <number of labels>)
        '''
        pass


class LabelLoaderHDF5(LabelLoader):

    def __init__(self, hdf5_path: str, bed_path: str):

        """
        Loads labels corresponding to the regions in a UCSC-BED file (bed_path).
        
        Labels are stored in an HDF5 file (hdf5_path). The file must contain two HDF5-datasets: "query" [dtype int, with shape (N,)] and "value" [dtype int, with shape (N,1)].
        The query corresponds to the 0-based position in the BED-file (i.e, a single region).
        
        A notebook that produces data in this format is "220601_process_label_data_hg38.ipynb"
        I later added attributes using "221111_add_dataset_attributes.ipynb"
        
        The "query"-data must be sorted in ascending order! Warning: No safety checks are perfomed.
        
        Data coding:
        
            Example:
            
            three queries (0,1,5) with one or more labels
            
            query   target
              0     3
              0     5
              0     8
              1     2
              1     3
              5     0
              
            corresponds to:
              
            query -> target
              0   -> [3,5,8]
              1   -> [2,3]
              5   -> [0]
            
            queries with no target labels must *not* be listed in the HDF5 file!
        
        
        Attributes:
        
            The 'md5' attribute of the 'query' dataset contains the md5-sum of the regions BED-file used to generate the data.
            The 'labels' attribute of the 'target' dataset contains the file identifiers (used for metadata matching).
            The 'N' attribute of the 'target' dataset contains the number of observations for every 'label' within the dataset.
            
            
            Example: 
            
                h5 = h5py.File(...,'r')
                h5['target'].attrs.get('labels')
        
        
        """

        with h5py.File(hdf5_path, 'r') as infile:
            # data us stored in memory
            queries = infile['query'][:]
            values = infile['target'][:, 0]

            self.label_ids = infile['target'].attrs.get('labels')
            self.label_N = infile['target'].attrs.get('N')
            self.bed_md5 = infile['query'].attrs.get('md5')

        assert len(queries) == len(
            values), f'Error: length of queries ({len(queries)}) does not match length of target values ({len(values)}) in {hdf5_path}'
        assert len(queries) > 0, f'Error: data has length 0 (empty HDF5 dataset)'

        if md5(bed_path) != self.bed_md5:
            print(
                f'Warning: the md5-sum of the supplied BED-file ({bed_path}) does not match the one stored in the hdf5 dataset attributes!')

        if bed_path.endswith('.gz'):
            with gzip.open(bed_path, 'rt') as infile:
                for i, _ in enumerate(infile):
                    pass
                bed_len = i + 1
                print(f'BED-file contains {bed_len} regions.')
        else:
            with open(bed_path, 'r') as infile:
                for i, _ in enumerate(infile):
                    pass
                bed_len = i + 1
                print(f'BED-file contains {bed_len} regions.')

        uniqval = np.unique(queries, return_index=True)

        assert np.max(uniqval[0]) < bed_len, f'Error: there are query keys that exceed the length of the BED-file'

        print('{:.3f}% of regions have at least 1 label.'.format(len(uniqval[0]) / bed_len * 100))

        self.lookup_array = np.empty((bed_len,), dtype=object)
        self.lookup_array[uniqval[0]] = np.split(values, uniqval[1][1:])  # this works only because the values in "query" are sorted!

        self.n_labels = len(self.label_ids)

        max_label = np.max(values) + 1
        assert self.n_labels <= max_label, f'Error: found {self.n_labels} labels in the hdf5 attributes, but found label with larger index ({max_label}) in the target data.'

    def __len__(self):
        return len(self.lookup_array)

    def __getitem__(self, i):
        return self.lookup_array[i]

    def get_labels(self, i):

        '''
        Return labels for region(s) "i" in sparse representation, e.g. [1,10,32,100] (or None if no labels are present)
        
        output will have shape (len(i), ?)
        '''

        return self.lookup_array[i]

    def get_label_ids(self):

        '''
        Return the label identifiers 
        '''
        return self.label_ids

    def get_labels_one_hot(self, i):

        '''
        Return labels for region(s) "i" in 1-hot representation
        
        output will have shape (len(i), self.n_labels)
        '''

        arr = self.get_labels(i)

        if arr is not None:
            res = np.zeros((len(arr), self.n_labels))
        else:
            res = np.zeros((1, self.n_labels))
            return res

        for i, l in enumerate(arr):
            if l is not None:
                res[i, l] = 1.

        return res
