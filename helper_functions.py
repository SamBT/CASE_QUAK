import numpy as np
import h5py

def smoothen_integers(input_dist):
    if (input_dist%1==0).all():
        print("integer distribution detected...")
        input_dist += np.random.normal(loc=0., scale=0.5, size=len(input_dist))
    return input_dist

def smoothen_padding(input_dist, padding_threshold=10, ref_neighbor=10):
    # padding_threshold: minimum number of entries that define a padded edge
    # ref_neighbor: distance to this neighbor defines width and position 
    
    lower_edge = min(input_dist)
    n_lower_edge = sum(input_dist==lower_edge)
    lower_padding = n_lower_edge>padding_threshold
    upper_edge = max(input_dist)
    n_upper_edge = sum(input_dist==upper_edge)
    upper_padding = n_upper_edge>padding_threshold

    if not (lower_padding or upper_padding):
        print("no padding detected...")
        return input_dist
    elif not upper_padding:
        print("lower padding detected...")
    elif not lower_padding:
        print("upper padding detected...")
    else:
        print("upper and lower padding detected...")

    unique_vals = np.unique(input_dist)

    if lower_padding:
        near_neighbor_pos = unique_vals[1]
        ref_neighbor_pos = unique_vals[ref_neighbor]
        edge_width = abs(ref_neighbor_pos - near_neighbor_pos)
        
        gauss_vals = np.random.normal(loc=near_neighbor_pos-edge_width,
            scale=edge_width, size=n_lower_edge)
        input_dist[input_dist==lower_edge]=gauss_vals

    if upper_padding:
        near_neighbor_pos = unique_vals[-2]
        ref_neighbor_pos = unique_vals[-ref_neighbor-1]
        edge_width = abs(ref_neighbor_pos - near_neighbor_pos)
        
        gauss_vals = np.random.normal(loc=near_neighbor_pos+edge_width,
            scale=edge_width, size=n_upper_edge)
        input_dist[input_dist==upper_edge]=gauss_vals

    return input_dist

def get_batch_file(inp_sample_type, inp_new_samples, inp_batch_number): 
    
    if not inp_new_samples: 
            batch = "/nobackup/users/myunus/CASE_samples/BB_batch%s.h5" % (inp_batch_number)
    else: 
        if inp_sample_type == 'QCDBKG': 
            batch = "/home/myunus/CASE_QCDBKG/qcdbkg_%s.h5" % (inp_batch_number)
        elif inp_sample_type == 'Wp2000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/WpToBpT_Wp2000_Bp25_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5"
        elif inp_sample_type == 'Wp3000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/WpToBpT_Wp3000_Bp170_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5"
        elif inp_sample_type == 'Wp5000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/WpToBpT_Wp5000_Bp400_Top170_Zbt_TuneCP5_13TeV-madgraphMLM-pythia8_TIMBER.h5"
        elif inp_sample_type == 'RSGraviton2000': 
             batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/RSGravitonToGluonGluon_kMpl01_M_2000_TuneCP5_13TeV_pythia8_TIMBER.h5"
        elif inp_sample_type == 'RSGraviton3000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/RSGravitonToGluonGluon_kMpl01_M_3000_TuneCP5_13TeV_pythia8_TIMBER.h5"
        elif inp_sample_type == 'RSGraviton5000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/RSGravitonToGluonGluon_kMpl01_M_5000_TuneCP5_13TeV_pythia8_TIMBER.h5"    
        elif inp_sample_type == 'Qstar2000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/QstarToQW_M_2000_mW_25_TuneCP2_13TeV-pythia8_TIMBER.h5"
        elif inp_sample_type == 'Qstar3000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/QstarToQW_M_3000_mW_170_TuneCP2_13TeV-pythia8_TIMBER.h5"
        elif inp_sample_type == 'Qstar5000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/QstarToQW_M_5000_mW_400_TuneCP2_13TeV-pythia8_TIMBER.h5"
        elif inp_sample_type == 'Wkk2000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/WkkToWRadionToWWW_M2000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5"
        elif inp_sample_type == 'Wkk3000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/WkkToWRadionToWWW_M3000_Mr170_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5"
        elif inp_sample_type == 'Wkk5000': 
            batch = "/home/wmccorma/CASE_Signal_Samples_Apr14/2018/WkkToWRadionToWWW_M5000_Mr400_TuneCP5_13TeV-madgraph-pythia8_TIMBER.h5"
        elif inp_sample_type == 'CMS': 
            if inp_batch_number < 100: 
                batch = "/home/wmccorma/CASE_SmallDataset_April21/Data_2018A_%s.h5" % (inp_batch_number + 900)
            elif inp_batch_number < 110: 
                batch = "/home/wmccorma/CASE_SmallDataset_April21/Data_2018A_%s.h5" % (inp_batch_number - 10)
            else: 
                batch = "/home/wmccorma/CASE_SmallDataset_April21/Data_2018A_9.h5"
    return batch
    
def LAPS_train(sample_type, num_batches=1, Mjj_cut=1200, pt_cut=550, eta_cut=None, inp_meanstd=None):    #LAPS stands for Load And Process Samples
    
    new_samples = False
    
    if sample_type == 'qcdbkg': 
        inp_truth_label = 0
    elif sample_type == 'graviton': 
        inp_truth_label = 1
        eta_cut = None
    elif sample_type == 'wprimesig': 
        inp_truth_label = 2
        eta_cut = None
    elif sample_type == 'Wkk': 
        inp_truth_label = 3
        eta_cut = None
    else: 
        new_samples = True
        if sample_type != 'QCDBKG': 
            num_batches = 1
        eta_cut = None

    data = np.array([])

    for batch_number in range(num_batches): 
        
        train_batch = get_batch_file(sample_type, new_samples, batch_number)
        f = h5py.File(train_batch, "r")
        
        jet_kinematics = f['jet_kinematics']
        jet1_extraInfo = f['jet1_extraInfo']
        jet2_extraInfo = f['jet2_extraInfo']
        truth_label = f['truth_label']

        np.seterr(invalid = 'ignore')

        delta_eta = jet_kinematics[:,1]

        Mjj = np.reshape(jet_kinematics[:,0], (-1,1))
        Mj1 = np.reshape(jet_kinematics[:,5], (-1,1))
        Mj2 = np.reshape(jet_kinematics[:,9], (-1,1))

        jet1_pt = np.reshape(jet_kinematics[:,2], (-1,1))
        jet2_pt = np.reshape(jet_kinematics[:,6], (-1,1))

        jet1_tau1 = np.reshape(jet1_extraInfo[:,0], (-1,1))
        jet1_tau2 = np.reshape(jet1_extraInfo[:,1], (-1,1))
        jet1_tau3 = np.reshape(jet1_extraInfo[:,2], (-1,1))
        jet1_tau4 = np.reshape(jet1_extraInfo[:,3], (-1,1))
        jet1_btagscore = np.reshape(jet1_extraInfo[:,5],(-1,1))
        jet1_numpfconst = np.reshape(jet1_extraInfo[:,6],(-1,1))

        jet1_tau21 = jet1_tau2 / jet1_tau1
        jet1_tau32 = jet1_tau3 / jet1_tau2
        jet1_tau43 = jet1_tau4 / jet1_tau3
        jet1_sqrt_tau21 = np.sqrt(jet1_tau21) / jet1_tau1

        jet2_tau1 = np.reshape(jet2_extraInfo[:,0], (-1,1))
        jet2_tau2 = np.reshape(jet2_extraInfo[:,1], (-1,1))
        jet2_tau3 = np.reshape(jet2_extraInfo[:,2], (-1,1))
        jet2_tau4 = np.reshape(jet2_extraInfo[:,3], (-1,1))
        jet2_btagscore = np.reshape(jet2_extraInfo[:,5],(-1,1))
        jet2_numpfconst = np.reshape(jet2_extraInfo[:,6],(-1,1))

        jet2_tau21 = jet2_tau2 / jet2_tau1
        jet2_tau32 = jet2_tau3 / jet2_tau2
        jet2_tau43 = jet2_tau4 / jet2_tau3
        jet2_sqrt_tau21 = np.sqrt(jet2_tau21) / jet2_tau1

        if not new_samples: 
            truth_label = truth_label[:]

        data = np.concatenate((Mj1, jet1_tau21, jet1_tau32, jet1_tau43, jet1_sqrt_tau21, jet1_btagscore, jet1_numpfconst,
                               Mj2, jet2_tau21, jet2_tau32, jet2_tau43, jet2_sqrt_tau21, jet2_btagscore, jet2_numpfconst), axis=1)

        if not new_samples: 
            indices = np.where((truth_label == inp_truth_label) 
                                      & (Mjj > Mjj_cut) 
                                      & (jet1_pt > pt_cut) 
                                      & (jet2_pt > pt_cut)
                                      & (np.isfinite(jet1_tau21))
                                      & (np.isfinite(jet1_tau32))
                                      & (np.isfinite(jet1_tau43))
                                      & (np.isfinite(jet1_sqrt_tau21))
                                      & (np.isfinite(jet2_tau21))
                                      & (np.isfinite(jet2_tau32))
                                      & (np.isfinite(jet2_tau43))
                                      & (np.isfinite(jet2_sqrt_tau21)))[0]
        else: 
            indices = np.where((Mjj > Mjj_cut) 
                                      & (jet1_pt > pt_cut) 
                                      & (jet2_pt > pt_cut)
                                      & (np.isfinite(jet1_tau21))
                                      & (np.isfinite(jet1_tau32))
                                      & (np.isfinite(jet1_tau43))
                                      & (np.isfinite(jet1_sqrt_tau21))
                                      & (np.isfinite(jet2_tau21))
                                      & (np.isfinite(jet2_tau32))
                                      & (np.isfinite(jet2_tau43))
                                      & (np.isfinite(jet2_sqrt_tau21)))[0]
            
        if eta_cut is not None: 
            eta_indices = np.where((np.abs(delta_eta) < eta_cut))[0]
            indices = np.intersect1d(indices, eta_indices)

        if batch_number == 0: 
            norm_data = data[indices]
            masses = Mjj[indices]
        else: 
            norm_data = np.concatenate((norm_data, data[indices]), axis=0)
            masses = np.concatenate((masses, Mjj[indices]), axis=0)

    #bad_jet1_btag_indices = np.where((data[:,5] < 0))[0]
    #bad_jet2_btag_indices = np.where((data[:,12] < 0))[0]

    #data[bad_jet1_btag_indices,5] = 0
    #data[bad_jet2_btag_indices,12] = 0

    #smooth_jet1_btagscore = smoothen_padding(data[:,5])
    #smooth_jet2_btagscore = smoothen_padding(data[:,12])

    smooth_jet1_numpfconst = smoothen_integers(norm_data[:,6])
    smooth_jet2_numpfconst = smoothen_integers(norm_data[:,13])

    unnorm_data = np.copy(norm_data)
    
    if inp_meanstd is None: 
        for index in range(norm_data.shape[1]):
            mean = np.mean(norm_data[:,index])
            std = np.std(norm_data[:,index])
            norm_data[:,index] = (norm_data[:,index]-mean)/std
    else: 
        mean_vec = inp_meanstd[0]
        std_vec = inp_meanstd[1]
        for index in range(norm_data.shape[1]): 
            mean = mean_vec[index]
            std = std_vec[index]
            norm_data[:,index] = (norm_data[:,index]-mean)/std
        
    return norm_data, unnorm_data, masses

def LAPS_test(sample_type, num_batches=1, Mjj_cut=1200, pt_cut=550, eta_cut=None, inp_meanstd=None):    #LAPS stands for Load And Process Samples
    
    new_samples = False
    
    if sample_type == 'qcdbkg': 
        inp_truth_label = 0
    elif sample_type == 'graviton': 
        inp_truth_label = 1
        eta_cut = None
    elif sample_type == 'wprimesig': 
        inp_truth_label = 2
        eta_cut = None
    elif sample_type == 'Wkk': 
        inp_truth_label = 3
        eta_cut = None
    elif sample_type == 'top_MB': 
        inp_truth_label = -1
    elif sample_type == 'ttbar_MB': 
        inp_truth_label = -2
    elif sample_type == 'vjets_MB': 
        inp_truth_label = -3
    elif sample_type == 'CMS': 
        print("Please use the LAPS_test_CMS function to test on CMS samples.")
        return None
    else: 
        new_samples = True
        if sample_type != 'QCDBKG': 
            num_batches = 1
        eta_cut = None

    data = np.array([])

    BNUB = 40    #BNUB stands for Batch Number Upper Bound
    if sample_type == 'QCDBKG': 
        BNUB = 12

    for batch_number in range(BNUB - num_batches, BNUB): 

        test_batch = get_batch_file(sample_type, new_samples, batch_number)
        f = h5py.File(test_batch, "r")

        jet_kinematics = f['jet_kinematics']
        jet1_extraInfo = f['jet1_extraInfo']
        jet2_extraInfo = f['jet2_extraInfo']
        truth_label = f['truth_label']

        np.seterr(invalid = 'ignore')

        delta_eta = jet_kinematics[:,1]

        Mjj = np.reshape(jet_kinematics[:,0], (-1,1))
        Mj1 = np.reshape(jet_kinematics[:,5], (-1,1))
        Mj2 = np.reshape(jet_kinematics[:,9], (-1,1))

        jet1_pt = np.reshape(jet_kinematics[:,2], (-1,1))
        jet2_pt = np.reshape(jet_kinematics[:,6], (-1,1))

        jet1_tau1 = np.reshape(jet1_extraInfo[:,0], (-1,1))
        jet1_tau2 = np.reshape(jet1_extraInfo[:,1], (-1,1))
        jet1_tau3 = np.reshape(jet1_extraInfo[:,2], (-1,1))
        jet1_tau4 = np.reshape(jet1_extraInfo[:,3], (-1,1))
        jet1_btagscore = np.reshape(jet1_extraInfo[:,5],(-1,1))
        jet1_numpfconst = np.reshape(jet1_extraInfo[:,6],(-1,1))

        jet1_tau21 = jet1_tau2 / jet1_tau1
        jet1_tau32 = jet1_tau3 / jet1_tau2
        jet1_tau43 = jet1_tau4 / jet1_tau3
        jet1_sqrt_tau21 = np.sqrt(jet1_tau21) / jet1_tau1

        jet2_tau1 = np.reshape(jet2_extraInfo[:,0], (-1,1))
        jet2_tau2 = np.reshape(jet2_extraInfo[:,1], (-1,1))
        jet2_tau3 = np.reshape(jet2_extraInfo[:,2], (-1,1))
        jet2_tau4 = np.reshape(jet2_extraInfo[:,3], (-1,1))
        jet2_btagscore = np.reshape(jet2_extraInfo[:,5],(-1,1))
        jet2_numpfconst = np.reshape(jet2_extraInfo[:,6],(-1,1))

        jet2_tau21 = jet2_tau2 / jet2_tau1
        jet2_tau32 = jet2_tau3 / jet2_tau2
        jet2_tau43 = jet2_tau4 / jet2_tau3
        jet2_sqrt_tau21 = np.sqrt(jet2_tau21) / jet2_tau1
        
        if not new_samples: 
            truth_label = truth_label[:]

        data = np.concatenate((Mj1, jet1_tau21, jet1_tau32, jet1_tau43, jet1_sqrt_tau21, jet1_btagscore, jet1_numpfconst,
                               Mj2, jet2_tau21, jet2_tau32, jet2_tau43, jet2_sqrt_tau21, jet2_btagscore, jet2_numpfconst), axis=1)

        if not new_samples: 
            indices = np.where((truth_label == inp_truth_label) 
                                      & (Mjj > Mjj_cut) 
                                      & (jet1_pt > pt_cut) 
                                      & (jet2_pt > pt_cut)
                                      & (np.isfinite(jet1_tau21))
                                      & (np.isfinite(jet1_tau32))
                                      & (np.isfinite(jet1_tau43))
                                      & (np.isfinite(jet1_sqrt_tau21))
                                      & (np.isfinite(jet2_tau21))
                                      & (np.isfinite(jet2_tau32))
                                      & (np.isfinite(jet2_tau43))
                                      & (np.isfinite(jet2_sqrt_tau21)))[0]
        else: 
            indices = np.where((Mjj > Mjj_cut) 
                                      & (jet1_pt > pt_cut) 
                                      & (jet2_pt > pt_cut)
                                      & (np.isfinite(jet1_tau21))
                                      & (np.isfinite(jet1_tau32))
                                      & (np.isfinite(jet1_tau43))
                                      & (np.isfinite(jet1_sqrt_tau21))
                                      & (np.isfinite(jet2_tau21))
                                      & (np.isfinite(jet2_tau32))
                                      & (np.isfinite(jet2_tau43))
                                      & (np.isfinite(jet2_sqrt_tau21)))[0]

        if eta_cut is not None: 
            eta_indices = np.where((np.abs(delta_eta) < eta_cut))[0]
            indices = np.intersect1d(indices, eta_indices)

        if batch_number == BNUB - num_batches: 
            norm_data = data[indices]
            masses = Mjj[indices]
        else: 
            norm_data = np.concatenate((norm_data, data[indices]), axis=0)
            masses = np.concatenate((masses, Mjj[indices]), axis=0)

    #bad_jet1_btag_indices = np.where((data[:,5] < 0))[0]
    #bad_jet2_btag_indices = np.where((data[:,12] < 0))[0]

    #data[bad_jet1_btag_indices,5] = 0
    #data[bad_jet2_btag_indices,12] = 0

    #smooth_jet1_btagscore = smoothen_padding(data[:,5])
    #smooth_jet2_btagscore = smoothen_padding(data[:,12])

    smooth_jet1_numpfconst = smoothen_integers(norm_data[:,6])
    smooth_jet2_numpfconst = smoothen_integers(norm_data[:,13])

    unnorm_data = np.copy(norm_data)
    
    if inp_meanstd is None: 
        for index in range(norm_data.shape[1]):
            mean = np.mean(norm_data[:,index])
            std = np.std(norm_data[:,index])
            norm_data[:,index] = (norm_data[:,index]-mean)/std
    else: 
        mean_vec = inp_meanstd[0]
        std_vec = inp_meanstd[1]
        for index in range(norm_data.shape[1]): 
            mean = mean_vec[index]
            std = std_vec[index]
            norm_data[:,index] = (norm_data[:,index]-mean)/std
        
    return norm_data, unnorm_data, masses

def LAPS_test_CMS(sample_type='CMS', num_batches=1, Mjj_cut=1200, pt_cut=550, eta_cut=None, inp_meanstd=None):    #LAPS stands for Load And Process Samples

    new_samples = True
    
    data = np.array([])

    for batch_number in range(num_batches): 

        test_batch = get_batch_file(sample_type, new_samples, batch_number)
        f = h5py.File(test_batch, "r")

        jet_kinematics = f['jet_kinematics']
        jet1_extraInfo = f['jet1_extraInfo']
        jet2_extraInfo = f['jet2_extraInfo']
        truth_label = f['truth_label']

        np.seterr(invalid = 'ignore')

        delta_eta = jet_kinematics[:,1]

        Mjj = np.reshape(jet_kinematics[:,0], (-1,1))
        Mj1 = np.reshape(jet_kinematics[:,5], (-1,1))
        Mj2 = np.reshape(jet_kinematics[:,9], (-1,1))

        jet1_pt = np.reshape(jet_kinematics[:,2], (-1,1))
        jet2_pt = np.reshape(jet_kinematics[:,6], (-1,1))

        jet1_tau1 = np.reshape(jet1_extraInfo[:,0], (-1,1))
        jet1_tau2 = np.reshape(jet1_extraInfo[:,1], (-1,1))
        jet1_tau3 = np.reshape(jet1_extraInfo[:,2], (-1,1))
        jet1_tau4 = np.reshape(jet1_extraInfo[:,3], (-1,1))
        jet1_btagscore = np.reshape(jet1_extraInfo[:,5],(-1,1))
        jet1_numpfconst = np.reshape(jet1_extraInfo[:,6],(-1,1))

        jet1_tau21 = jet1_tau2 / jet1_tau1
        jet1_tau32 = jet1_tau3 / jet1_tau2
        jet1_tau43 = jet1_tau4 / jet1_tau3
        jet1_sqrt_tau21 = np.sqrt(jet1_tau21) / jet1_tau1

        jet2_tau1 = np.reshape(jet2_extraInfo[:,0], (-1,1))
        jet2_tau2 = np.reshape(jet2_extraInfo[:,1], (-1,1))
        jet2_tau3 = np.reshape(jet2_extraInfo[:,2], (-1,1))
        jet2_tau4 = np.reshape(jet2_extraInfo[:,3], (-1,1))
        jet2_btagscore = np.reshape(jet2_extraInfo[:,5],(-1,1))
        jet2_numpfconst = np.reshape(jet2_extraInfo[:,6],(-1,1))

        jet2_tau21 = jet2_tau2 / jet2_tau1
        jet2_tau32 = jet2_tau3 / jet2_tau2
        jet2_tau43 = jet2_tau4 / jet2_tau3
        jet2_sqrt_tau21 = np.sqrt(jet2_tau21) / jet2_tau1
        
        if not new_samples: 
            truth_label = truth_label[:]

        data = np.concatenate((Mj1, jet1_tau21, jet1_tau32, jet1_tau43, jet1_sqrt_tau21, jet1_btagscore, jet1_numpfconst,
                               Mj2, jet2_tau21, jet2_tau32, jet2_tau43, jet2_sqrt_tau21, jet2_btagscore, jet2_numpfconst), axis=1)

        if not new_samples: 
            indices = np.where((truth_label == inp_truth_label) 
                                      & (Mjj > Mjj_cut) 
                                      & (jet1_pt > pt_cut) 
                                      & (jet2_pt > pt_cut)
                                      & (np.isfinite(jet1_tau21))
                                      & (np.isfinite(jet1_tau32))
                                      & (np.isfinite(jet1_tau43))
                                      & (np.isfinite(jet1_sqrt_tau21))
                                      & (np.isfinite(jet2_tau21))
                                      & (np.isfinite(jet2_tau32))
                                      & (np.isfinite(jet2_tau43))
                                      & (np.isfinite(jet2_sqrt_tau21)))[0]
        else: 
            indices = np.where((Mjj > Mjj_cut) 
                                      & (jet1_pt > pt_cut) 
                                      & (jet2_pt > pt_cut)
                                      & (np.isfinite(jet1_tau21))
                                      & (np.isfinite(jet1_tau32))
                                      & (np.isfinite(jet1_tau43))
                                      & (np.isfinite(jet1_sqrt_tau21))
                                      & (np.isfinite(jet2_tau21))
                                      & (np.isfinite(jet2_tau32))
                                      & (np.isfinite(jet2_tau43))
                                      & (np.isfinite(jet2_sqrt_tau21)))[0]

        if eta_cut is not None: 
            eta_indices = np.where((np.abs(delta_eta) < eta_cut))[0]
            indices = np.intersect1d(indices, eta_indices)

        if batch_number == 0: 
            norm_data = data[indices]
            masses = Mjj[indices]
        else: 
            norm_data = np.concatenate((norm_data, data[indices]), axis=0)
            masses = np.concatenate((masses, Mjj[indices]), axis=0)

    #bad_jet1_btag_indices = np.where((data[:,5] < 0))[0]
    #bad_jet2_btag_indices = np.where((data[:,12] < 0))[0]

    #data[bad_jet1_btag_indices,5] = 0
    #data[bad_jet2_btag_indices,12] = 0

    #smooth_jet1_btagscore = smoothen_padding(data[:,5])
    #smooth_jet2_btagscore = smoothen_padding(data[:,12])

    smooth_jet1_numpfconst = smoothen_integers(norm_data[:,6])
    smooth_jet2_numpfconst = smoothen_integers(norm_data[:,13])

    unnorm_data = np.copy(norm_data)
    
    if inp_meanstd is None: 
        for index in range(norm_data.shape[1]):
            mean = np.mean(norm_data[:,index])
            std = np.std(norm_data[:,index])
            norm_data[:,index] = (norm_data[:,index]-mean)/std
    else: 
        mean_vec = inp_meanstd[0]
        std_vec = inp_meanstd[1]
        for index in range(norm_data.shape[1]): 
            mean = mean_vec[index]
            std = std_vec[index]
            norm_data[:,index] = (norm_data[:,index]-mean)/std
        
    return norm_data, unnorm_data, masses