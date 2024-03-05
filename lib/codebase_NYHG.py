# Codebase -- NYHG CCEP dataset
import numpy as np
import pylab as plt
import sys
sys.path.insert(0, 'C:\\Users\\esolo\\Documents\\Python Scripts\\lib')

# Times recorded in index along raw array
# good_channels refers to the channel indices in the raw ecog_labels field that are used for analysis
resting_database = {'Pt8': {'prestim_resting': [], 
                            'poststim_resting': [],
                           },
                    'Pt10': {'prestim_resting':[3285,170234], 
                             'poststim_resting':[1140271,1315115],
                             'good_channels':[range(0, 63), range(64, 109)],
                             'depths': ['LDIP*', 'LDIA*'],
                             'grids': ['PMT*', '2 GRID*', '3 GRID*'],
                             'stim_label': ['2 GRID8', '2 GRID13']
                            },
                   'Pt13': {'prestim_resting': [], 
                           'poststim_resting': []
                           }}


class subject_connectivity():
    
    def __init__(self, subject, root='D:/NYHG_repStim/'):
        
        # Initialize some variables
        self.subject = subject
        self.root = root
        
        # For parallel processing
        import multiprocessing as mp
        self.n_cpus = mp.cpu_count()
        
    def load_data(self):
        '''
        Loads raw data (including resting periods) from .mat file. For now, we assume
        this file is already re-referenced (bipolar for depths, and common average across all grids/strips)
        '''
        
        import mat73
        self.raw = mat73.loadmat(self.root+self.subject+'/NYHG_'+self.subject+'_raw.mat')['raw']
        self.srate = int(self.raw['srate'])
        self.prepost = mat73.loadmat(self.root+self.subject+'/NYHG_'+self.subject+'_prepost.mat')['dataAll']
        self.raw_channels = self.raw['ecog_labels']
        
    def visualize_raw_trace(self, chan=None, start=0, stop=-1):
        '''
        View the entire stream of averaged data in an interactive plot, or an individual electrode.
        '''
        import pylab as plt
        plt.figure(figsize=(11, 2)); ax=plt.subplot(111)
        
        if chan is None:
            plt.plot(np.mean(self.raw['data'][:, start:stop], 0), linewidth=0.5)
        else:
            plt.plot(self.raw['data'][chan, start:stop], linewidth=0.5)
            
        return
    
    def set_prestim_resting_data(self, start, winsize, numwins, dsample=None):
        '''
        Establish a period of time for resting-state pre-stimulation data from which to compute functional connectivity.
        
        start: time to begin resting period (in seconds)
        winsize: window in which to compute connectivity (in seconds)
        numwins: number of windows to average over
        dsample (optional): Factor by which to downsample the data (integer)
        '''
        if self.reref==True:
            data = self.reref_data
        else:
            data = self.raw['data']   # use the raw array if no rereferenced data exists
        
        # Restructure the data into windows x channels x samples, for easier processing with MNE. 
        dat = []
        self.srate = self.raw['srate']  # need to re-set the sample rate if revisiting the raw data
        for i in range(data.shape[0]):
            a = np.reshape(data[i, int(start*self.srate):int(start*self.srate)+int(self.srate*numwins*winsize)], 
                          (numwins, int(winsize*self.srate)))
            dat.append(a[:, np.newaxis, :])
        dat = np.concatenate(dat, axis=1)
         
        # Optional: downsample the data    
        if dsample is not None:
            from mne.filter import resample
            dat_dsample = resample(dat.astype(float), down=dsample, npad='auto')
            self.prestim_resting = dat_dsample
            self.srate = int(self.srate/dsample)
            print('New sampling rate is '+str(self.srate)+' Hz')
            return dat_dsample
        else:
            self.prestim_resting = dat
            return dat
    
        
    def compute_connectivity(self, data, freqs, method, n_cycles, mode, plot_=True, verbose=False, return_average=True, n_jobs=None):
        
        import mne_connectivity
        
        if n_jobs is None:
            n_jobs = self.n_cpus
        else:
            pass
        
        # Compute connectivities
        con = mne_connectivity.spectral_connectivity_time(data,
                                                 freqs = freqs, 
                                                 method = method, n_cycles = n_cycles,
                                                 sfreq = self.srate, 
                                                 mode = mode, 
                                                 n_jobs=n_jobs, verbose=verbose)
        
        # Create adjacency matrix
        adj = con.get_data(output='dense')
        if return_average==False:
            return adj
        else:
            adj = np.mean(np.mean(adj, 3), 0)
            adj = adj+adj.T
        
        # Plot adjacency matrix
        if plot_==True:
            plt.matshow(adj, cmap='inferno', vmin=0.0, vmax=1.0)
            plt.colorbar()
            plt.tight_layout()
        else:
            pass
        
        return adj
    
    def drop_unused_elecs_old(self, adj):
        '''
        Some electrodes (usually near the end of the montage) are not used but their channel IDs remain in the dataset.
        '''
        meta = self.prepost['metaData']
        stimChan = meta['stimChan']
        chanLabels = meta['chanlabels']  # list of channels used for analysis
        ecogLabels = self.raw['ecog_labels']
        
        # Only use the preserved channels to construct a reduced matrix
        chans2drop = []
        for idx, e in enumerate(ecogLabels):
            if e not in chanLabels:
                chans2drop.append(idx)
                
        # Include any indices which do not have proper electrode locations
        r, c = np.where(np.isnan(self.raw['ecog_coords']))
        chans2drop.extend(list(np.unique(r)))
        chans2drop = np.unique(chans2drop)
        
        # Copy the array
        from copy import copy
        adj_reduced = copy(adj)

        for c in np.sort(chans2drop)[::-1]:  # deleting the unused rows/columns
            adj_reduced = np.delete(adj_reduced, c, axis=0)
            adj_reduced = np.delete(adj_reduced, c, axis=1)
            
        self.dropped_chans = chans2drop
        return adj_reduced
    
    def drop_unused_channels(self):
        '''
        Best to remove bad channels prior to other operations
        '''
        gchans = np.concatenate([list(i) for i in resting_database[self.subject]['good_channels']])
        self.good_channels = np.array(mysub.raw_channels)[gchans][:, 0]
        
        return
    
    def ca_bipolar_rereference(self, reref_channels):
        '''
        Takes a channel list and performs common average and bipolar referencing, inferred from the structure of the array elements.
        Hyphenated elements are considered bipolar. Grids/strips assumed to be grouped according to the resting_database entry.
        '''
        from copy import copy
    
        # Copy the un-rereferenced data
        #raw_data_cp = copy(self.raw['data'])  # slow step that isn't really necessary
        reref_data = np.empty([len(reref_channels), self.raw['data'].shape[1]]); reref_data[:] = np.nan

        # Find the channel groupings
        groups = resting_database[self.subject]['grids']

        # First rereference the common average channels, sorted by group
        for g in groups:
            g = g[:-1]
            idxs = np.where(np.char.find(reref_channels.astype(str), g)!=-1)[0]
            mu = np.mean(self.raw['data'][idxs, :], axis=0)
            reref_data[idxs, :] = self.raw['data'][idxs, :]-mu

        bp_pairs = np.where(np.char.find(reref_channels.astype(str), '-')!=-1)[0]
        for c in bp_pairs:

            # Identify the channels
            chn_name = reref_channels[c]
            anode = chn_name.split('-')[0]
            cathode = chn_name.split('-')[1]
            anode_idx = np.where(np.array(self.raw_channels)==[anode])[0][0]
            cathode_idx = np.where(np.array(self.raw_channels)==[cathode])[0][0]

            # Do the bipolar substraction
            reref_data[c, :] = self.raw['data'][anode_idx, :]-self.raw['data'][cathode_idx, :]

        self.reref_channels = reref_channels
        self.reref_data = reref_data
        self.reref = True
        print('Re-referenced data set!')
        return
    
    def residualize_distance(self, adj, transformation=None):
        '''
        Optional: Pass a transformation (logit, log, or fisher) to normalize the FC data. 
        '''
        
        # Remove the rows which correspond to unusued channels (usually coded as NaNs anyway
        coords = self.raw['ecog_coords']
        coords = np.delete(coords, self.dropped_chans, axis=0)
        
        # Construct a Euclidean distance adjacency matrix
        from utils import construct_distance_adjacency
        self.dist_adj = construct_distance_adjacency(coords)
        
        ### Residualize connecitivty based on distances ###
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        try:
            # Construct vectors for linear regression
            dist_vec = self.dist_adj[np.triu_indices_from(self.dist_adj, 1)].reshape(-1, 1)
            fc_vec = adj[np.triu_indices_from(adj, 1)].reshape(-1, 1)
            if dist_vec.size!=fc_vec.size:
                print("!!! vectors are not the same size...check your adjacency matrices.")
                return None
            
            if transformation is None:
                fc_vec = fc_vec
            elif transformation=='logit':
                from scipy.special import logit
                fc_vec = logit(fc_vec)
            elif transformation=='log':
                fc_vec = np.log10(fc_vec)
            elif transformation=='fisher':
                fc_vec = np.arctanh(fc_vec)
            else:
                pass

            model.fit(dist_vec, fc_vec)
            fc_predict = model.predict(dist_vec)
            fc_resid = fc_vec-fc_predict
        
        except:
            print("Encountered an error!")
            return
        
        # Re-broadcast the modified connectivities into an adjacency matrix
        adj_resid = np.empty(adj.shape); adj_resid[:] = np.nan; 
        adj_resid[np.triu_indices_from(adj_resid, 1)] = fc_resid.ravel()
        
        return adj_resid
        
        
        