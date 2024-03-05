#Codebase -- TMS/EEG
import pandas as pd
import numpy as np

def construct_bipolar_montage(d):
    """
    d is a Dataframe containing ONLY depth electrodes in a given subject.
    """
    all_anodes = []; all_cathodes = []
    
    unique_groups = d['Group'].unique()
    for g in unique_groups:
        for side_ in ['Left', 'Right', 'Unknown']:
            
            # Check for the hemisphere before constructing bipolar pairs
            sub_array = d[(d['Group']==g) & (d['Side']==side_)]
            if len(sub_array)==0:
                continue
                
            group_anodes = ['LFPx'+str(i) for i in sub_array['Channel'][:-1]]
            group_cathodes = ['LFPx'+str(i) for i in sub_array['Channel'][1:]]
            all_anodes.extend(group_anodes)
            all_cathodes.extend(group_cathodes)
        
    return all_anodes, all_cathodes

def remove_bad_chans(bad_chans, ch_names):
    """
    Identifies the bad channels to remove, including from bipolar pairs
    """
    
    # Remove bad electrodes
    chans_to_drop = []
    for c in ch_names:
        ex = False
        monopols = c.split('-')
        for m in monopols:
            for b in bad_chans:
                b_str = 'LFPx'+b
                if b_str==m:
                    ex = True
        if ex:
            chans_to_drop.append(c)
    return chans_to_drop

def reref_contacts_frame(contacts, channels):
    """
    Returns a contacts DataFrame corrected for bipolar rereferencing. Inherits localization info from
    the anode, and averages the MNI and avg coordinates. 
    """
    from copy import copy
    new_contacts = pd.DataFrame()
    for c in channels:
        monopols = c.split('-')
        if len(monopols)==1:
            # this wasn't a bipolar electrode -- just keep it as it! 
            entry = contacts[contacts['Channel']==int(monopols[0][4:])]
            entry['Channel'] = c
            new_contacts = new_contacts.append(entry, ignore_index=True)
        else:
            # this WAS a bipolar electrode
            c1 = contacts[contacts['Channel']==int(monopols[0][4:])]
            c2 = contacts[contacts['Channel']==int(monopols[1][4:])]

            # inherit the localization info from the anode, average the position
            entry = copy(c1)
            entry['Channel'] = c
            entry['mniX'] = (c1['mniX'].iloc[0]+c2['mniX'].iloc[0])/2.; entry['mniY'] = (c1['mniY'].iloc[0]+c2['mniY'].iloc[0])/2.; entry['mniZ'] = (c1['mniZ'].iloc[0]+c2['mniZ'].iloc[0])/2.
            entry['anatX'] = (c1['anatX'].iloc[0]+c2['anatX'].iloc[0])/2.; entry['anatY'] = (c1['anatY'].iloc[0]+c2['anatY'].iloc[0])/2.; entry['anatZ'] = (c1['anatZ'].iloc[0]+c2['anatZ'].iloc[0])/2.
            new_contacts = new_contacts.append(entry, ignore_index=True)
    return new_contacts

def signalblender(a, b):
    """
    Blends two signals (should be equal legnth) by ramping down the amplitude of one as the other ramps up. 
    Should only do with short snippets to ensure relative stationarity.
    """
    a_weight = np.linspace(1, 0, num=a.size, endpoint=True)
    b_weight = np.linspace(0, 1, num=b.size, endpoint=True)
    
    # Apply weights to the signals
    a_weighted = a*a_weight
    b_weighted = b*b_weight
    
    # Combine the signals
    return a_weighted+b_weighted

def interpolate_stim(eeg, trigTimes, sfreq):
    """
    Uses signalblender to interpolate EEG during the stim artifact. Operates on one channel at a time.
    Scrubs -25ms to +25ms around stim artifact with 50ms blended, mirrored signals from the immediately preceeding and following EEG. 
    """
    from copy import copy
    eeg = copy(eeg)  # since we are modifying in-place
    
    for t in trigTimes: # for each stim pulse
        
        # Get the 50ms mirrored signals (buffered by 25 ms from the stim artifact itself)
        before = eeg[t-int(sfreq*0.075):t-int(sfreq*0.025)][::-1]
        after = eeg[t+int(sfreq*0.025):t+int(sfreq*0.075)][::-1]
        
        # Blend the signals
        blended = signalblender(before, after)
        
        # Replace the artifact interval
        eeg[t-int(sfreq*0.025):t+int(sfreq*0.025)] = blended
        
    return eeg
        

def matlabToPython(eng, content, key_name):
    '''
    Converts the 'unreadable' MATLAB table objects into readable pandas dataframes
    
    eng: MATLAB engine instance
    fname: loaded .mat file via MATLAB engine
    keyname: key which points to the data object within the .mat file you wish to load
    '''
    
    eng.workspace["table"] = content[key_name]
    jsontable = eng.eval('jsonencode(table)')
    return pd.read_json(jsontable)

def normalize_prestim(arr_, samples=250): 
    '''
    docstring here
    '''
    
    from copy import copy
    arr = copy(arr_)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            mu = np.mean(arr[i, j, 0:samples])
            std_ = np.std(arr[i, j, 0:samples])
            arr[i, j, :] = (arr[i, j, :]-mu)/std_
    return arr

def get_window_power(dat, fs, start_time, win_size, freqs, n_cpus):
    '''
    docstring here
    '''
    
    from mne.time_frequency import psd_array_multitaper
    pow_, freqs_used = psd_array_multitaper(dat[:, :, int(start_time*fs):int(start_time*fs)+int(fs*win_size)], 
                                                    sfreq=fs, fmin = freqs[0], fmax=freqs[-1], 
                                                    output='power', verbose=False, n_jobs=n_cpus)
    pow_ = np.mean(np.log10(pow_), 2)
    
    return pow_, freqs_used

def get_window_power_epo(dat, start_time, end_time, freqs, n_cpus, favg=True):
    '''
    Same as above but for an EpochsArray
    '''
    pow_, freqs_used = dat.compute_psd(method='multitaper', fmin=freqs[0], fmax=freqs[-1], tmin=start_time, tmax=end_time, 
                                verbose=False, n_jobs=n_cpus).get_data(return_freqs=True)
    
    if favg==True:
        pow_ = np.mean(np.log10(pow_), 2)
    else: 
        pow_ = np.log10(pow_)
    
    return pow_, freqs_used

def get_saturated_elecs(eeg, samples=250, thresh_=10):
    '''
    Takes an EEG array of shape trials x electrodes x samples and identifies likely amplifier saturation. 
    
    Measures TEP and excluses any electrode with a mean TEP z-score greater than 10 (default) -- highly correlated with amplifier saturation (and a rather conservative threshold). Returns electrode indices. 
    
    samples: number of samples from the start of the array to use as normalization baseline
    thresh_: if a different threshold is desired
    '''
    
    # Normalize relative to pre-stim interval
    eeg_norm = normalize_prestim(eeg, samples=samples)
    
    # Average normalized waveforms to construct the TEPs
    eeg_norm_mu = np.nanmean(eeg_norm, axis=0)
    
    # Filter electrodes w/ TEP z-score greater than 10 (highly correlated with amplifier saturation)
    saturated_elecs = np.unique(np.where(np.abs(eeg_norm_mu)>thresh_)[0])
    
    return saturated_elecs

def remove_bad_epochs(eeg_, thresh=10, channel_remove=0.2): 
    '''
    Takes an events x channels x time array of EEG data and identifies any eventxchannel pair
    with an amplitude value that exceeds a z-score threshold relative to the rest of the data
    
    Returns an EEG array with the bad epochs as NaN, as well as a list of the event x channel pairs that were removed.
    
    Removes an entire channel if more than channel_remove of events for that channel are bad. 
    '''
    
    from copy import copy
    eeg = copy(eeg_)
    
    # Z-score the data
    mu = np.nanmean(eeg.ravel())
    std_ = np.nanstd(eeg.ravel())
    eeg_norm = (eeg-mu)/std_
    
    # Remove any channel/epoch pair which contain any sample w/ Z>thresh
    eeg_norm = np.abs(eeg_norm)
    ev_idx, chan_idx, _ = np.where(eeg_norm>thresh)
    
    # There will be many useless duplicates
    l = list(zip(ev_idx, chan_idx))
    unique_list = list(dict.fromkeys(l))
    
    for t in unique_list:
        eeg[t[0], t[1], :] = np.nan
    print(str(len(unique_list))+' event/channel pairs removed.')
    
    perc_removed = np.sum(np.isnan(np.sum(eeg, 2)), 0)/eeg.shape[0]
    channels2remove = np.where(perc_removed>channel_remove)[0]
    for c in channels2remove:
        eeg[:, c, :] = np.nan
        print('Removed channel '+str(c)+' due to excess noise.')
    
    return eeg, unique_list, channels2remove

class subject_tfr:
    '''
    docstring here
    '''
    
    def __init__(self, fs=500, root='D:/'):
        import numpy as np
        import pandas as pd

        self.fs = fs  #sampling frequency
        self.root = root

        # For parallel processing
        import multiprocessing as mp
        self.n_cpus = mp.cpu_count()
    
    def load_data(self, sub, sess):
        
        # Load the data
        self.tms_dat = np.load('./TMSEEG_data/'+sub+'/'+sub+'_TMS_'+sess+'_0_5Hz_1-CleanedFilteredSubsampled.npy')
        self.sham_dat = np.load('./TMSEEG_data/'+sub+'/'+sub+'_Sham_'+sess+'_0_5Hz_1-CleanedFilteredSubsampled.npy')  
        self.sub = sub
        self.sess = sess
        
    def load_data_epo(self, sub, sess):
        
        from mne import read_epochs
        from glob import glob
        
        fname = self.root+str(sub)+'/complete_epo.fif'
        epochs = read_epochs(fname, preload=True)
        epochs = epochs['target.str.contains("'+sess+'")']
        
        #Set TMS/sham data
        #self.tms_dat = epochs['tms'].get_data()
        #self.sham_dat = epochs['sham'].get_data()
        self.epochs = epochs
        
        dirs = glob(self.root+str(sub)+'/*/')
        self.elecs = pd.read_pickle(dirs[0]+'contacts.pkl') 
        self.sub = sub
        self.sess = sess

    def set_params(self, n_cycles=None):
        # Set some params
        self.freqs = np.logspace(np.log10(4), np.log10(110), num=25)
        if n_cycles is None:
            self.n_cycles = [2, 2, 2, 2,
                    2, 2, 3, 3,
                    3, 3, 3, 3, 
                    4, 4, 4, 4, 
                    4, 4, 5, 5, 
                    5, 5, 5, 5, 5]  # pre-specify this to allow for best time-frequency resolution tradeoff
        else:
            self.n_cycles = n_cycles

    def set_tfr_power(self, picks=None, mirror_size=None): 
        import mne
        
        if picks is not None:
            picked_channels = self.epochs.pick_channels(picks)
            tms_dat = picked_channels['tms'].get_data()
            sham_dat = picked_channels['sham'].get_data()
        else:
            tms_dat = self.epochs['tms'].get_data()
            sham_dat = self.epochs['sham'].get_data()
        
        if mirror_size is not None: 
            #Add mirror buffers (better than nothing but can induce some spectral changes)
            tms_dat = np.concatenate((np.flip(tms_dat[:, :, :mirror_size], axis=2), tms_dat, np.flip(tms_dat[:, :, -mirror_size:], axis=2)), axis=2)  #Using 450ms buffers
            sham_dat = np.concatenate((np.flip(sham_dat[:, :, :mirror_size], axis=2), sham_dat, np.flip(sham_dat[:, :, -mirror_size:], axis=2)), axis=2)
            
        
        tfr_tms = mne.time_frequency.tfr_array_morlet(tms_dat, sfreq=self.epochs.info['sfreq'],
                                                          freqs=self.freqs, output='power',
                                                          n_cycles=self.n_cycles,
                                                          n_jobs=self.n_cpus) # epochs, chans, freqs, times
        tfr_tms = np.log10(tfr_tms)

        tfr_sham = mne.time_frequency.tfr_array_morlet(sham_dat, sfreq=self.epochs.info['sfreq'],
                                                          freqs=self.freqs, output='power',
                                                          n_cycles=self.n_cycles,
                                                          n_jobs=self.n_cpus) # epochs, chans, freqs, times
        tfr_sham = np.log10(tfr_sham)
        
        if mirror_size is not None: 
            # Clip the buffers
            self.tfr_tms = tfr_tms[:, :, :, mirror_size:-mirror_size]
            self.tfr_sham = tfr_sham[:, :, :, mirror_size:-mirror_size]
        else:
            self.tfr_tms = tfr_tms
            self.tfr_sham = tfr_sham
    
    def set_elecs(self):
        self.elecs = pd.read_pickle('./TMSEEG_data/'+self.sub+'/'+self.sub+'_'+self.sess+'_elecs.pkl')
        
    def subtract_baseline(self, start_, fin):
        
        # May want to consider subtracting off the mean power from the pre-stimulation period to actually measure CHANGE in power
        for i in range(self.tfr_tms.shape[0]):
            for j in range(self.tfr_tms.shape[1]):
                for k in range(self.tfr_tms.shape[2]):
                    self.tfr_tms[i, j, k, :] = self.tfr_tms[i, j, k, :]-np.mean(self.tfr_tms[i, j, k, start_:fin]) 

        for i in range(self.tfr_sham.shape[0]):
            for j in range(self.tfr_sham.shape[1]):
                for k in range(self.tfr_sham.shape[2]):
                    self.tfr_sham[i, j, k, :] = self.tfr_sham[i, j, k, :]-np.mean(self.tfr_sham[i, j, k, start_:fin])
                    
    def set_t_statistic(self, e):
        from scipy.stats import ttest_ind
        self.t, self.p = ttest_ind(self.tfr_tms[:, e, :, :], self.tfr_sham[:, e, :, :], axis=0, equal_var=False)
        
    def plot_tfr(self, data, figsize=(13, 5), vmin=3.5, vmax=3.5, fontsize=14):
        
        import pylab as plt
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        
        plt.matshow(data, fignum=0, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        plt.yticks(np.arange(0, len(self.freqs), 4), np.round(self.freqs[::4]))
        plt.xticks(ax.get_xticks(), np.round((ax.get_xticks()/self.fs)-0.5, 3)); ax.xaxis.set_ticks_position('bottom')
        ax.invert_yaxis()
        plt.vlines([0.5*self.fs], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color='black', linestyle='--', linewidth=3)
        plt.title('Time-frequency decomposition', fontsize=fontsize); plt.xlabel('Time (sec)', fontsize=fontsize); plt.ylabel('Frequency (Hz)', fontsize=fontsize)
        cbar = plt.colorbar(); cbar.set_label('T-stat')
        
        
    def compute_ERP(self, norm_samples=250):
        
        from scipy.stats import sem
        from mne.filter import filter_data
        
        # Filter between 1Hz and 35Hz for ERP analysis
        tms_dat = filter_data(self.epochs['tms'].get_data(), self.fs, l_freq=1, h_freq=35, verbose=False)
        sham_dat = filter_data(self.epochs['sham'].get_data(), self.fs, l_freq=1, h_freq=35, verbose=False)
        
        # Normalize against baseline (defined as number of samples from onset of EEG clip) 
        tms_norm = normalize_prestim(tms_dat, samples=norm_samples); 
        sham_norm = normalize_prestim(sham_dat, samples=norm_samples); 
        
        # Get the ERPs and SEMs
        self.sep_tms_mu = np.mean(tms_norm, 0)
        self.sep_sham_mu = np.mean(sham_norm, 0)
        self.sep_tms_sem = sem(tms_norm, axis=0)
        self.sep_sham_sem = sem(sham_norm, axis=0)
        
    def plot_ERP(self, channel_name):
        
        channel_idx = np.where(np.array(self.epochs.ch_names)==channel_name)[0][0]
        
        return self.sep_tms_mu[channel_idx, :], self.sep_tms_sem[channel_idx, :], self.sep_sham_mu[channel_idx, :], self.sep_sham_sem[channel_idx, :]



# Subjects with reasonably intact data and corresponding stim locations/sessions
good_subs = {'404':['STG', 'Motor'],
             '405':['STG', 'Parietal'], 
             '416':['Parietal', 'Parietal1', 'STG'],
             '423':['Parietal'],
             '429':['L_DLPFC'],
             '430':['Precuneus', 'Parahippocampus', 'DLPFC'],
             '477':['Parietal'],
             '483':['Parietal', 'DLPFC'],
             '518':['DLPFC'],
             '534':['DLPFC'],
             '538':['DLPFC'],
             '559':['DLPFC'],
             '561':['DLPFC']}

bad_subs = {'460': ['DLPFC'], '477': ['DLPFC']} #need some cleanup work but files are available

# Get stim locations in native xyz space, channel number indicated in the last column if available
# Note that the stim location information is incomplete, not well documented, and stitched together from multiple sources. Will need to be reconciled for publishable analysis.
# Stim locations are stored in "ECoG_TMS_Summary.xls" The 5XX subject stim locations come from RAS coordinates in "PatientFileLocs" (assuming 0.5Hz and ESTT sites are the same)
target_dict = {'404': {'STG': [54.74, -39.25, -19.73, np.nan], 'Motor': [42.53, -67.34, 8.96, np.nan]}, 
                '405': {'STG': [-63.33, -21.23, 18.35, 252], 'Parietal': [39.67, -78.83, 25.35, 46]}, 
              '416': {'Parietal': [24.09, -75.34, 40.37, 23], 'Parietal1': [24.09, -75.34, 40.37, 23], 'STG': [64.09, -24.94, -2.63, 206]}, 
              '423': {'Parietal': [22.81, -89.17, 30.29, 88]},
               '429': {'L_DLPFC': [-38.35, 31.46, 23.01, np.nan]}, 
               '430': {'Precuneus': [-12.74, -53.56, 54.19, 70], 'Parahippocampus': [0.26, -76.76, 29.19, 97], 'DLPFC': [-42.74, 34.44, 28.19, np.nan]},
               '477': {'Parietal': [-34.19, -85.56, 2.82, 55]},
               '483': {'Parietal': [-46.16, -76.82, 22.41, np.nan], 'DLPFC': [-35.53, 4.71, 53.16, np.nan]}, 
               '518': {'DLPFC': [-39.7, 57.23, 27.97, np.nan]}, 
               '534': {'DLPFC': [-26.36, 66.65, 11.43, np.nan]}, 
               '538': {'DLPFC': [-36.26, 44.13, 23.12, np.nan]}, 
               '559': {'DLPFC': [35.16, 45.15, 37.25, np.nan]}, 
               '561': {'DLPFC': [-42.83, 11.55, 36.00, 228]}}

# Anatomic filters
filter_dict = {'l_frontal_filter': '^Ctx-lh.*(frontal|parsopercu|parsorbi|paracentral|precentral|parstriang)',
'r_frontal_filter': '^Ctx-rh.*(frontal|parsopercu|parsorbi|paracentral|precentral|parstriang)',
'l_temporal_filter': '^Ctx-lh.*(temporal|fusiform|bankssts)',
'r_temporal_filter': '^Ctx-rh.*(temporal|fusiform|bankssts)',
'l_mtl_filter': '(^Ctx-lh.*(entorhinal|parahipp))|(^Left.*Hippocampus)',
'r_mtl_filter': '(^Ctx-rh.*(entorhinal|parahipp))|(^Right.*Hippocampus)',
'l_parietal_filter': '^Ctx-lh.*(supramarginal|parietal|cuneus)',
'r_parietal_filter': '^Ctx-rh.*(supramarginal|parietal|cuneus)',
'l_limbic_filter': '(^Ctx-lh.*(entorhinal|insula|cingulate|parahipp))|(^Left.*(Hippo|Amygd))',
'r_limbic_filter': '(^Ctx-rh.*(entorhinal|insula|cingulate|parahipp))|(^Right.*(Hippo|Amygd))',
'frontal_filter': 'frontal|parsopercu|parsorbi|paracentral|precentral|parstriang',
'temporal_filter': 'temporal|fusiform|bankssts', 
'parietal_filter': 'supramarginal|parietal|cuneus',
'mtl_filter': 'entorhinal|parahipp|Hippo',
'limbic_filter': 'entorhinal|insula|cingulate|parahipp|Hippo|Amygd',
'hippocampus_filter': 'Hippo', 'r_hippocampus_filter':'Right-Hippo', 'l_hippocampus_filter': 'Left-Hippo',
'entorhinal_filter': 'entorhinal', 'r_entorhinal_filter': 'rh-entorhinal', 'l_entorhinal_filter': 'lh-entorhinal', 
'parahippocampal_filter': 'parahipp', 'r_parahippocampal_filter': 'rh-parahipp', 'l_parahippocampal_filter': 'lh-parahipp',
'amygdala_filter': 'Amyg', 'l_amygdala_filter': 'Left-Amyg', 'r_amygdala_filter': 'R-Amyg'}

lobe_dict = {'frontal': ['ctx-rh-lateralorbitofrontal', 'ctx-lh-lateralorbitofrontal',
       'ctx-lh-parsorbitalis', 'ctx-lh-precentral',
       'ctx-lh-caudalmiddlefrontal', 'ctx-lh-parsopercularis',
       'ctx-lh-paracentral', 'ctx-lh-rostralmiddlefrontal',
       'ctx-rh-medialorbitofrontal', 'ctx-lh-medialorbitofrontal',
       'ctx-lh-superiorfrontal', 'ctx-lh-parstriangularis',
       'ctx-rh-paracentral', 'ctx-rh-rostralmiddlefrontal',
       'ctx-rh-precentral', 'ctx-rh-parstriangularis',
       'ctx-rh-parsorbitalis', 'ctx-rh-superiorfrontal',
       'ctx-rh-caudalmiddlefrontal', 'ctx-rh-parsopercularis'], 
             
            'limbic': ['right-amygdala', 'left-amygdala', 'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-insula', 'ctx-rh-insula', 
       'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal', 'left-hippocampus', 'right-hippocampus',
       'ctx-lh-rostralanteriorcingulate', 'ctx-lh-caudalanteriorcingulate', 'ctx-lh-isthmuscingulate', 'ctx-lh-posteriorcingulate',
       'ctx-rh-rostralanteriorcingulate', 'ctx-rh-caudalanteriorcingulate', 'ctx-rh-isthmuscingulate', 'ctx-rh-posteriorcingulate',],
            
            'temporal': ['ctx-lh-bankssts', 'ctx-lh-fusiform', 'ctx-lh-inferiortemporal', 
                        'ctx-lh-lingual', 'ctx-lh-middletemporal', 'ctx-lh-superiortemporal', 
                        'ctx-lh-inferiortemporal', 'ctx-lh-temporalpole', 'ctx-lh-transversetemporal', 
                        'ctx-rh-bankssts', 'ctx-rh-fusiform', 'ctx-rh-inferiortemporal', 
                        'ctx-rh-lingual', 'ctx-rh-middletemporal', 'ctx-rh-superiortemporal', 
                        'ctx-rh-inferiortemporal', 'ctx-rh-temporalpole', 'ctx-rh-transversetemporal'], 
             
            'parietal': ['ctx-lh-inferiorparietal', 'ctx-lh-superiorparietal', 'ctx-lh-supramarginal', 'ctx-lh-postcentral', 'ctx-lh-precuneus', 'ctx-lh-cuneus', 'ctx-rh-inferiorparietal', 'ctx-rh-superiorparietal', 'ctx-rh-supramarginal', 'ctx-rh-postcentral', 'ctx-rh-precuneus', 'ctx-rh-cuneus'], 
            
            'mtl': ['left-hippocampus', 'right-hippocampus', 'ctx-lh-entorhinal', 'ctx-rh-entorhinal', 'ctx-lh-parahippocampal', 'ctx-rh-parahippocampal']} 



