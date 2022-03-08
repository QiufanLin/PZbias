import numpy as np
import tensorflow as tf
import time
import pickle
from PZbias_network import *
import threading
import argparse
from scipy.optimize import least_squares



####################
##### Settings #####
####################


##### 'directory_output' is only used for testing the code. 
##### All the saved models and results are under 'directory_input'; be careful not to delete/overwrite them.

directory_main = '/renoir_data_02/qiufan/'
directory_input = directory_main + 'photoz/'
directory_output = directory_main + 'photoz/test/'
directory_SDSSdata = '/renoir_data_02/jpasquet/data_redshift/'
index1_ini = np.load(directory_main + 'random_seed_1000000.npy')
index2_ini = np.load(directory_main + 'random_seed_393219.npy')
index3_ini = np.load(directory_main + 'random_seed_102547.npy')
index_gauss_ini = np.load(directory_main + 'random_gauss_seed_1000000.npy')


parser = argparse.ArgumentParser()
parser.add_argument('--ne', help='No. Expriment', type=int)
parser.add_argument('--fth', help='flattening threshold', type=int)
parser.add_argument('--nsub', help='num of subtrain', type=int)
parser.add_argument('--errl', help='error added to labels', type=float)
parser.add_argument('--midreduce', help='reduce (middle) training set', type=int)
parser.add_argument('--testphase', help='if test', type=int)
parser.add_argument('--tstep', help='training step', type=int)
parser.add_argument('--usecfht', help='use CFHT', type=int)
parser.add_argument('--usecfhtd', help='use CFHT DEEP', type=int)
parser.add_argument('--bins', help='bins', type=int)
parser.add_argument('--net', help='network', type=int)
parser.add_argument('--itealter', help='iteration of alter checkpoint', type=int)
parser.add_argument('--softlabel', help='use soft labels (step 3)', type=int)
parser.add_argument('--shiftlabel', help='shift labels (step 3)', type=int)

args = parser.parse_args()
num_expri = args.ne
fth = args.fth
nsub = args.nsub
errl = args.errl
midreduce = args.midreduce
testphase = args.testphase
tstep = args.tstep
usecfht = args.usecfht
usecfhtd = args.usecfhtd
bins = args.bins
net = args.net
itealter = args.itealter
softlabel = args.softlabel
shiftlabel = args.shiftlabel



##### Iterations, learning rate, training batch sizes, data, expriment settings #####


if tstep < 2:
    iterations = 120000  # total number of iterations for training
    ite_point_save = [40000, 80000, 120000]  # save model at number of iterations during training
    learning_rate_step = iterations / 2
else:
    iterations = 40000
    ite_point_save = [800]  #[40000]
    learning_rate_step = 1000000  


learning_rate_ini = 0.0001
lr_reduce_factor = 5.0
iterations_pre = 120000
ite_test_print = 200  # print test results per number of iterations during training


if usecfht == 0 and tstep < 2:
    batch_train = 128  #8  #16  #32  #64  #128
else:
    batch_train = 32
    

imag_size = 64
channels = 5
NormMethod = 2
cross_val = 1

repetition_per_ite = 1
use_cpu = False
clip_grad = False
num_threads = 16+1
batch_th = int(batch_train/(num_threads-1)) + int(2 * (batch_train/(num_threads-1) - int(batch_train/(num_threads-1))))

use_SDSS = False
use_CFHTmix = False
use_CFHTmW = False
use_CFHTmD = False

if usecfht == 0:
    use_SDSS = True
elif usecfht == 1:
    use_CFHTmix = True
    num_reserve = 10000   #20000
    if tstep > 1 and usecfhtd == 1: use_CFHTmD = True
    if tstep > 1 and usecfhtd == 0: use_CFHTmW = True


pretrain = False
parttrain = False
if tstep > 1:
    pretrain = True
    parttrain = True
    
if pretrain:
    learning_rate_ini = learning_rate_ini / 5.0

test_phase = False
save_train = False   
if testphase == 1:
    test_phase = True
    if tstep > 1 and not (tstep == 3 and use_CFHTmW):
        save_train = True



###### Redshift #####


if use_CFHTmix:
    z_min = 0.0
    z_max = 4.0
    wbin = (z_max - z_min) / bins
    z_prix = 'z4_bin' + str(bins)

elif use_SDSS:
    z_min = 0.0
    z_max = 0.4
    wbin = (z_max - z_min) / bins
    z_prix = 'z04'
    if bins != 180: z_prix = z_prix + '_bin' + str(bins)

z_prix = z_prix + '_cv' + str(cross_val) + 'ne' + str(num_expri)


zrange_ext = False
ydisp_shift = False
if tstep == 3: 
    zrange_ext = True
    ydisp_shift = True
    
if zrange_ext: 
    bins_l_ext = bins / 2
    if use_CFHTmix: bins_r_ext = bins
    elif use_SDSS: bins_r_ext = bins / 2
    
    if bins == 20:
        bins_l_ext = 20
        bins_r_ext = 20
    bins = bins + bins_l_ext + bins_r_ext
    zl_ext = bins_l_ext * wbin
    print ('zext:', z_min, z_max, zl_ext, bins_l_ext, bins_r_ext, bins, wbin)
else: 
    bins_l_ext = 0
    bins_r_ext = 0
    zl_ext = 0
    


###### r-band magnitude #####
   
 
multi_r_out = False
if tstep > 0: multi_r_out = True
if use_CFHTmix: num_multi_r_out = 10
elif use_SDSS: num_multi_r_out = 6 

even_sampling_permag = False
if tstep > 1: even_sampling_permag = True     
flat_threshold = fth   #2  #5   #10  #None
     
if even_sampling_permag or multi_r_out:
    if use_CFHTmix:
        r_min = 16.0
        r_max = 25.5
        rbins = 19
        rwbin = (r_max - r_min) / rbins  #0.5
    elif use_SDSS:
        r_min = 12.5   #13.5   #14.0
        r_max = 18.0
        rbins = 11   #9    #8
        rwbin = (r_max - r_min) / rbins  #0.5
     
     
     
###### Variants #####


midreduce_train_SDSS = False
if use_SDSS and midreduce == 1: midreduce_train_SDSS = True
midreduce_left_train_SDSS = 1 
midreduce_right_train_SDSS = 1
ratio_mid_train_SDSS = 0.1


alter_checkpoint = False 
ite_altercheckpoint = 0 
if use_SDSS and itealter != 0: 
    alter_checkpoint = True
    ite_altercheckpoint = itealter   #e.g., 5000
    if tstep < 2: 
        iterations_pre = 240000
        learning_rate_step = 1000000
        ite_point_save = [2000, 5000, 10000, 20000, 30000, 60000, 90000, 120000, 160000, 200000, 240000]


sub_train_SDSS = False
if use_SDSS and nsub > 0: sub_train_SDSS = True
num_sub_train_SDSS = nsub
even_sampling_no_subtrain = True


use_errlabel = False
if use_SDSS and errl > 0: use_errlabel = True
errlabel_sigma = errl
errlabel_frac = 1  #0.5
even_sampling_no_errlabel = True



###### Model load / save paths #####


if net == 0: Net = 'netJ_'
elif net == 1: Net = 'netT_'
elif net > 1: Net = 'net' + str(net - 1) + '_'
  
  
if use_CFHTmix:
    TrainData = 'trainCmixNewHQvLR2_resWD' + str(num_reserve) + '_'
    if use_CFHTmW: TrainData = TrainData.replace('Cmix', 'CmixW')
    if use_CFHTmD: TrainData = TrainData.replace('Cmix', 'CmixD')
elif use_SDSS: 
    TrainData = 'trainS2_'
    if midreduce_train_SDSS: 
        TrainData = TrainData + 'MidReduce' + str(ratio_mid_train_SDSS).replace('.', '') + '_'
        if midreduce_left_train_SDSS != 1: TrainData = TrainData + 'left' + str(midreduce_left_train_SDSS).replace('.', '') + '_'
        if midreduce_right_train_SDSS != 1: TrainData = TrainData + 'right' + str(midreduce_right_train_SDSS).replace('.', '') + '_'
    if sub_train_SDSS: 
        TrainData = TrainData + 'subTrain' + str(num_sub_train_SDSS) + '_'
        if even_sampling_permag and even_sampling_no_subtrain: TrainData = TrainData + 'noSub_'


Algorithm = 'ADAM_'
if repetition_per_ite != 1: Algorithm = Algorithm + 'rep' + str(repetition_per_ite) + '_'
if batch_train != 32: Algorithm = Algorithm + 'batch' + str(batch_train) + '_'
if iterations != 60000: Algorithm = Algorithm + 'ite' + str(iterations) + '_'
  
if alter_checkpoint and tstep > 1:
     Algorithm = Algorithm + 'iteAlter' + str(ite_altercheckpoint) + '_'
      
if use_errlabel: 
     Algorithm = Algorithm + ('errlabelFrac' + str(errlabel_frac) + 'Sig' + str(errlabel_sigma) + '_').replace('.', '')
     if even_sampling_permag and even_sampling_no_errlabel: Algorithm = Algorithm + 'noErr_'
                    
if tstep == 3: 
     Algorithm = Algorithm + 'yDisp_commonroutsigz_zExt' + str(bins_l_ext) + 'v' + str(bins_r_ext) + '_yDispExt_cmass2_dilated5mm_'
     if softlabel == 0: Algorithm = Algorithm + 'hard_'
     if shiftlabel == 0: Algorithm = Algorithm + 'noshift_'
            
if multi_r_out: 
     Algorithm = Algorithm + 'multiROut' + str(num_multi_r_out) + '_'
            

if even_sampling_permag:
    flat_term = 'flatth' + str(flat_threshold).replace('.', '') + '_'
    Sampling = 'evensamplepermag_' + flat_term
else: Sampling = ''
    

if pretrain:
    if use_CFHTmix:
        Pretrain = 'preCFHTmix_'
        model_load = 'model_T6_' + Net + 'nonrotate_' + Algorithm.replace('ite40000', 'ite' + str(iterations_pre)).replace('_iteAlter' + str(ite_altercheckpoint), '').replace('yDisp_commonroutsigz_zExt' + str(bins_l_ext) + 'v' + str(bins_r_ext) + '_yDispExt_', '').replace('noErr_', '').replace('cmass2_', '').replace('dilated5mm_', '').replace('hard_', '').replace('noshift_', '') + TrainData.replace('CmixW', 'Cmix').replace('CmixD', 'Cmix') + 'scratch_NoD_hardClabel_NonRegrid_' + z_prix + '_/'
        print ('pretrained_model_load:', model_load)
    elif use_SDSS:
        Pretrain = 'preSDSS_'
        model_load = 'model_T6_' + Net + 'nonrotate_' + Algorithm.replace('ite40000', 'batch128_ite' + str(iterations_pre)).replace('_iteAlter' + str(ite_altercheckpoint), '').replace('yDisp_commonroutsigz_zExt' + str(bins_l_ext) + 'v' + str(bins_r_ext) + '_yDispExt_', '').replace('noErr_', '').replace('cmass2_', '').replace('dilated5mm_', '').replace('hard_', '').replace('noshift_', '') + TrainData.replace('noSub_', '') + 'scratch_NoD_hardClabel_NonRegrid_' + z_prix + '_/'
        print ('pretrained_model_load:', model_load)
    if parttrain: Pretrain = Pretrain + 'parttrain_lastfc_'
else: Pretrain = 'scratch_'
    

ImgRandomRotate = 'nonrotate_'
Label = 'hardClabel_'
UseD = 'NoD_'
Regrid = 'NonRegrid_'


fx = Net + ImgRandomRotate + Algorithm + TrainData + Pretrain + Sampling + UseD + Label + Regrid + z_prix + '_'

model_savepath = directory_output + 'model_T6_' + fx + '/'
z_savepath = directory_output + 'probas_T6_' + fx
if tstep == 3: z_savepath = z_savepath.replace('T6_', '').replace('netJ_', '').replace('nonrotate_ADAM_', '').replace('_NoD_hardClabel_NonRegrid', '')
if test_phase and alter_checkpoint and tstep < 2: z_savepath = z_savepath + 'ite' + str(ite_altercheckpoint) + '_'
print ('#####') 
print (fx)
print ('#####') 


if test_phase:
#    model_load = model_savepath
    model_load = directory_input + 'model_T6_' + fx + '/'
    iterations = 0
elif tstep > 1:
    model_load = directory_input + model_load


###############################
##### Load Data & Process #####
###############################


##### Image normalization #####


def img_norm(img, NormMethod=1):
    if NormMethod == 0: pass   #no normalization
    elif NormMethod == 1: img = img / np.max(abs(img))   #normalized by the global peak flux
    elif NormMethod == 2:
        indice_neg = img < 0
        indice_pos = img >= 0
        img[indice_pos] = np.sqrt(img[indice_pos] + 1.0) - 1.0
        img[indice_neg] = -np.sqrt(-img[indice_neg] + 1.0) + 1.0   #sqrt of flux
    elif NormMethod == 3:
        img = img / np.max(abs(img), axis=(0,1))   #normalized by the peak flux of each band
    elif NormMethod == 4:
        img = img / 10    
    return img
 

##### CFHT data #####


if use_CFHTmix:
    imgs_ini = []  
    z_ini = []
    ebv_ini = []
    zphot_ini = []
    r_ini = []
    rerr_ini = []
    i_ini = []
    ierr_ini = []
    flag_ini = []
    field_ini = []
        
    for i in range(20):
        fi = np.load(directory_main + 'CUBE_WD_ZHQ_' + ('000' + str(i+1))[-4:] + '.npz')
        info = fi['info']
        cube = fi['cube']
        z_ini.append(info['ZSPEC'])
        ebv_ini.append(info['EBV'])
        zphot_ini.append(info['zphot'])
        r_ini.append(info['R'])
        rerr_ini.append(info['Rerr'])
        i_ini.append(info['I'])
        ierr_ini.append(info['Ierr'])
        flag_ini.append(info['ZFLAG'])
        field_ini.append(info['FIELD'])
        imgs_ini.append(cube)
                        
    z_ini = np.concatenate(z_ini)
    ebv_ini = np.concatenate(ebv_ini)
    zphot_ini = np.concatenate(zphot_ini)
    r_ini = np.concatenate(r_ini)
    rerr_ini = np.concatenate(rerr_ini)
    i_ini = np.concatenate(i_ini)
    ierr_ini = np.concatenate(ierr_ini)
    flag_ini = np.concatenate(flag_ini)
    field_ini = np.concatenate(field_ini)
    imgs_ini = np.concatenate(imgs_ini) / 1000.0    
    obj_all = np.arange(len(z_ini))[z_ini < z_max]
        
    obj_W = np.array([obj for obj in obj_all if 'W' in field_ini[obj]])
    obj_D = np.array([obj for obj in obj_all if 'D' in field_ini[obj]])
    obj_resW = obj_W[index1_ini[index1_ini < len(obj_W)]][:num_reserve]
    obj_resD = obj_D[index1_ini[index1_ini < len(obj_D)]][:num_reserve]
    obj_trainW = obj_W[index1_ini[index1_ini < len(obj_W)]][num_reserve:]
    obj_trainD = obj_D[index1_ini[index1_ini < len(obj_D)]][num_reserve:]
            
    if use_CFHTmW:
        obj_train = obj_trainW[index1_ini[index1_ini < len(obj_trainW)]]
        obj_test = obj_resW[index1_ini[index1_ini < len(obj_resW)]]
    elif use_CFHTmD:
        obj_train = obj_trainD[index1_ini[index1_ini < len(obj_trainD)]]
        obj_test = obj_resD[index1_ini[index1_ini < len(obj_resD)]]
    else:
        obj_train = np.concatenate([obj_trainW, obj_trainD], 0)[index1_ini[index1_ini < len(obj_trainW)+len(obj_trainD)]]
        obj_test = np.concatenate([obj_resW, obj_resD], 0)
    
    if tstep < 2:
        imgs_ini_add = []  
        z_ini_add = []
        ebv_ini_add = []
        zphot_ini_add = []
        r_ini_add = []
        rerr_ini_add = []
        i_ini_add = []
        ierr_ini_add = []
        flag_ini_add = []
        field_ini_add = []
        for i in range(2):
            fi = np.load(directory_main + 'CUBE_WD_ZLR_' + ('000' + str(i+1))[-4:] + '.npz')
            info = fi['info']
            cube = fi['cube']
            z_ini_add.append(info['ZSPEC'])
            ebv_ini_add.append(info['EBV'])
            zphot_ini_add.append(info['ZPHOT'])
            r_ini_add.append(info['R'])
            rerr_ini_add.append(info['ERRR'])
            i_ini_add.append(info['Ip_2'])
            ierr_ini_add.append(info['ERRIp_2'])
            flag_ini_add.append(info['ZFLAG'])
            field_ini_add.append(info['FIELD'])
            imgs_ini_add.append(cube)
                        
        z_ini_add = np.concatenate(z_ini_add)
        ebv_ini_add = np.concatenate(ebv_ini_add)
        zphot_ini_add = np.concatenate(zphot_ini_add)
        r_ini_add = np.concatenate(r_ini_add)
        rerr_ini_add = np.concatenate(rerr_ini_add)
        i_ini_add = np.concatenate(i_ini_add)
        ierr_ini_add = np.concatenate(ierr_ini_add)
        flag_ini_add = np.concatenate(flag_ini_add)
        field_ini_add = np.concatenate(field_ini_add)
        imgs_ini_add = np.concatenate(imgs_ini_add) / 1000.0
        obj_add = np.arange(len(z_ini_add))[z_ini_add < z_max] + len(z_ini)

        imgs_ini = np.concatenate([imgs_ini, imgs_ini_add])
        z_ini = np.concatenate([z_ini, z_ini_add])
        ebv_ini = np.concatenate([ebv_ini, ebv_ini_add])
        zphot_ini = np.concatenate([zphot_ini, zphot_ini_add])
        r_ini = np.concatenate([r_ini, r_ini_add])
        rerr_ini = np.concatenate([rerr_ini, rerr_ini_add])
        i_ini = np.concatenate([i_ini, i_ini_add])
        ierr_ini = np.concatenate([ierr_ini, ierr_ini_add])
        flag_ini = np.concatenate([flag_ini, flag_ini_add])
        field_ini = np.concatenate([field_ini, field_ini_add])
        obj_train = np.concatenate([obj_train, obj_add])
        
    print ('Read CFHT metadata')  
    if tstep < 2: print ('W,D,LR:', len(obj_W), len(obj_D), len(obj_add))        
    else: print ('W,D:', len(obj_W), len(obj_D))
    print ('Flags:', set(flag_ini))
    print ('Training,Test:', len(obj_train), len(obj_test))
    print ('#####') 


            
##### SDSS data #####


if use_SDSS:
    metadata = pickle.load(open(directory_SDSSdata + 'label', 'rb'))
    z_ini = metadata['z']
    ebv_ini = metadata['EBV']
    zphot_ini = metadata['zphot']
    r_ini = metadata['dered_petro_r'] #+ metadata['extinction_r']
    rerr_ini = metadata['petroMagErr_r']
    i_ini = metadata['dered_petro_i'] #+ metadata['extinction_i']
    ierr_ini = metadata['petroMagErr_i']   

    obj_train = np.load(directory_main + 'train.npy')
    obj_val = np.load(directory_main + 'val.npy')
    obj_test = np.load(directory_main + 'test.npy')
    
    err_index = np.argwhere(obj_test == 281531)
    obj_test = np.delete(obj_test, err_index)
    
    if sub_train_SDSS and not (even_sampling_permag and even_sampling_no_subtrain) and not test_phase:
        obj_train = obj_train[index2_ini[index2_ini < len(obj_train)]][:num_sub_train_SDSS]

    if midreduce_train_SDSS:
        zmid_l = 0.08
        zmid_r = 0.12
        z_train = z_ini[obj_train]
        obj_train_left = obj_train[z_train < zmid_l]
        obj_train_right = obj_train[z_train > zmid_r]
        obj_train_mid = obj_train[(z_train >= zmid_l) & (z_train <= zmid_r)]
        obj_train_left = obj_train_left[index2_ini[index2_ini < len(obj_train_left)]][:int(midreduce_left_train_SDSS*len(obj_train_left))]
        obj_train_right = obj_train_right[index2_ini[index2_ini < len(obj_train_right)]][:int(midreduce_right_train_SDSS*len(obj_train_right))]
        obj_train_mid = obj_train_mid[index2_ini[index2_ini < len(obj_train_mid)]][:int(ratio_mid_train_SDSS*len(obj_train_mid))]
        obj_train = np.concatenate([obj_train_left, obj_train_right, obj_train_mid])
     
    print ('Read SDSS metadata')            
    print ('Training,Test,Val:', len(obj_train), len(obj_test), len(obj_val))
    print ('#####')       
     
    

##### Remove NaN in magnitude #####
           

if multi_r_out:
    r_ini[np.isnan(r_ini)] = np.max(r_ini)
    r_ini[r_ini < 0] = np.max(r_ini)
       


##### Load test images & labels #####
 

if not test_phase: 
    obj_test = obj_test[:1000]
    
z_test = z_ini[obj_test]
ebv_test = ebv_ini[obj_test]                                
zphot_test = zphot_ini[obj_test]
r_test = r_ini[obj_test]                                
rerr_test = rerr_ini[obj_test]
i_test = i_ini[obj_test]
ierr_test = ierr_ini[obj_test] 

start = time.time()

if use_CFHTmix:
    imgs_test = imgs_ini[obj_test]        
    for i in range(len(imgs_test)):
        imgs_test[i] = img_norm(imgs_test[i], NormMethod)
    
if use_SDSS:
    imgs_test = np.zeros((len(obj_test), imag_size, imag_size, channels))   
    for i in range(len(obj_test)):
        img = np.load(directory_SDSSdata + 'data/' + str(obj_test[i]) + '.py.npy') 
        imgs_test[i] = img_norm(img, NormMethod)
               
y_test = np.zeros((len(obj_test), bins))
for i in range(len(obj_test)):
    z_index = max(0, min(bins - 1, int((z_test[i] - z_min) / wbin))) + bins_l_ext
    y_test[i, z_index] = 1.0

if multi_r_out:
    yr_test = np.zeros((len(obj_test), rbins))        
    for i in range(len(obj_test)):
        index_r_in = max(0, min(rbins - 1, int((r_test[i] - r_min) / rwbin)))
        yr_test[i, index_r_in] = 1.0
    if ydisp_shift: y_test = np.concatenate([yr_test] + num_multi_r_out * [y_test], 1)
    else: y_test = np.concatenate([yr_test, y_test], 1)
    
print ('Load test images & labels:', time.time() - start, 'seconds')
print ('#####') 



##### Add labeling errors to the SDSS training set (if True) #####


if use_SDSS and use_errlabel and not (even_sampling_permag and even_sampling_no_errlabel) and not test_phase:
    z_train = z_ini[obj_train]
    errlabel_index = np.arange(len(z_train))[index1_ini[index1_ini < len(z_train)]][:int(errlabel_frac*len(z_train))]
    z_train[errlabel_index] = z_train[errlabel_index] + errlabel_sigma * index_gauss_ini[:len(z_train[errlabel_index])]

    obj_train = obj_train[(z_train > z_min) & (z_train < z_max)]
    print ('errlabel:', str(len(obj_train)) +  '/' + str(len(z_train[errlabel_index])) + '/' + str(len(z_train)))



##### Flatten the training distribution for Steps 2 & 3 #####


if even_sampling_permag and (not test_phase or save_train):          
    z_train = z_ini[obj_train]
    zi_train = np.zeros(len(obj_train))
    for i in range(len(obj_train)):
        zi_train[i] = max(0, min(bins-bins_l_ext-bins_r_ext - 1, int((z_train[i] - z_min) / wbin)))           

    r_train = r_ini[obj_train]
    ri_train = np.zeros(len(obj_train))
    for i in range(len(obj_train)):
        ri_train[i] = max(0, min(rbins - 1, int((r_train[i] - r_min) / rwbin)))           

    index_select_train = []
    print ('subset_select:', num_expri)
    for i in range(bins-bins_l_ext-bins_r_ext):
        for j in range(rbins):
            index_ij = np.arange(len(obj_train))[(zi_train == i) & (ri_train == j)]
            if len(index_ij) > 0 and len(index_ij) <= flat_threshold: index_select_train.append(index_ij)
            elif len(index_ij) > 0: 
                index_to_select = np.concatenate([index_ij[index3_ini[index3_ini < len(index_ij)]]] * 5)
                index_select_train.append(index_to_select[(num_expri-1)*flat_threshold:num_expri*flat_threshold])
    index_select_train = np.concatenate(index_select_train, 0)        
    print ('flat_th:', flat_threshold, str(len(obj_train[index_select_train])) + '/' + str(len(obj_train)))  
    obj_train = obj_train[index_select_train]



##### Load training images & labels for saving (if True) #####


if save_train and test_phase and use_SDSS and len(obj_train) > 60000:
    num_save = 60000
    obj_train = obj_train[index1_ini[index1_ini < len(obj_train)][:num_save]]

z_train = z_ini[obj_train]
ebv_train = ebv_ini[obj_train]
zphot_train = zphot_ini[obj_train]
r_train = r_ini[obj_train]
rerr_train = rerr_ini[obj_train]
i_train = i_ini[obj_train]
ierr_train = ierr_ini[obj_train]

if save_train and test_phase:
    start = time.time()
    
    if use_CFHTmix:
        imgs_train = imgs_ini[obj_train]        
        for i in range(len(imgs_train)):
            imgs_train[i] = img_norm(imgs_train[i], NormMethod)
        
    if use_SDSS:
        imgs_train = np.zeros((len(obj_train), imag_size, imag_size, channels))   
        for i in range(len(obj_train)):
            img = np.load(directory_SDSSdata + 'data/' + str(obj_train[i]) + '.py.npy') 
            imgs_train[i] = img_norm(img, NormMethod)
                   
    y_train = np.zeros((len(obj_train), bins))
    for i in range(len(obj_train)):
        z_index = max(0, min(bins - 1, int((z_train[i] - z_min) / wbin))) + bins_l_ext
        y_train[i, z_index] = 1.0
    
    if multi_r_out:
        yr_train = np.zeros((len(obj_train), rbins))        
        for i in range(len(obj_train)):
            index_r_in = max(0, min(rbins - 1, int((r_train[i] - r_min) / rwbin)))
            yr_train[i, index_r_in] = 1.0
        if ydisp_shift: y_train = np.concatenate([yr_train] + num_multi_r_out * [y_train], 1)
        else: y_train = np.concatenate([yr_train, y_train], 1)
        
    print ('Load training images & labels:', time.time() - start, 'seconds')
    print ('#####')
    
    
    
##### Fit labels (softening & shifting) for Step 3 #####


if ydisp_shift and not test_phase:
    f_sub = directory_input + 'probas_T6_' + Net + 'nonrotate_' + Algorithm.replace('yDisp_commonroutsigz_zExt' + str(bins_l_ext) + 'v' + str(bins_r_ext) + '_yDispExt_', '').replace('cmass2_', '').replace('dilated5mm_', '').replace('hard_', '').replace('noshift_', '') + TrainData + Pretrain + 'evensamplepermag_' + flat_term + 'NoD_hardClabel_NonRegrid_' + z_prix + '_.npz'
    pre_est_sub = np.load(f_sub)
    print ('subset with pre-estimates:', f_sub)
    z_train_sub = pre_est_sub['z_train']
    r_train_sub = pre_est_sub['r_train']
    p_train_sub = pre_est_sub['probas_train']
        
    r_train_sub[np.isnan(r_train_sub)] = np.max(r_train_sub)
    r_train_sub[r_train_sub < 0] = np.max(r_train_sub)

    zlist = (np.arange(bins-bins_l_ext-bins_r_ext) + 0.5) * wbin
    zilist = np.arange(bins-bins_l_ext-bins_r_ext)
    
    zphoto_train_sub = np.zeros(len(p_train_sub))
    for i in range(len(p_train_sub)):
        zphoto_train_sub[i] = zlist[np.argmax(p_train_sub[i])]
       
    ri_train_sub = np.zeros(len(r_train_sub))
    for i in range(len(r_train_sub)):
        ri_train_sub[i] = max(0, min(rbins - 1, int((r_train_sub[i] - r_min) / rwbin)))
        
    dist2d_zr_train_sub = np.histogram2d(z_train_sub, ri_train_sub, (bins-bins_l_ext-bins_r_ext, rbins), ((z_min, z_max), (-0.5, rbins-0.5)))[0] / float(len(z_train_sub))
    dist2d_zphotor_train_sub = np.histogram2d(zphoto_train_sub, ri_train_sub, (bins-bins_l_ext-bins_r_ext, rbins), ((z_min, z_max), (-0.5, rbins-0.5)))[0] / float(len(z_train_sub))
    dist2d_zr_train_norm = np.clip(dist2d_zr_train_sub, 0, flat_threshold) / float(flat_threshold)

    
    ##### (a) Determining the labeling dispersions #####
    
    rbins_compress = int((rbins - 1) / num_multi_r_out)  #1  #2
    xdata = np.zeros((bins-bins_l_ext-bins_r_ext, num_multi_r_out))
    ydata = np.zeros((bins-bins_l_ext-bins_r_ext, num_multi_r_out))

    for j in range(rbins):  #interlacing
        index_r_out = min(max(0, int(j / (1 + rbins_compress))), num_multi_r_out - 1)
        xdata[:, index_r_out] = xdata[:, index_r_out] + dist2d_zphotor_train_sub[:, j]
        ydata[:, index_r_out] = ydata[:, index_r_out] + dist2d_zr_train_sub[:, j]
        if j > 0 and (j + 1) % (1 + rbins_compress) == 0:
            xdata[:, index_r_out+1] = xdata[:, index_r_out+1] + dist2d_zphotor_train_sub[:, j]
            ydata[:, index_r_out+1] = ydata[:, index_r_out+1] + dist2d_zr_train_sub[:, j]
    xdata = xdata / (np.sum(xdata, 0, keepdims=True) + 10**(-20))
    ydata = ydata / (np.sum(ydata, 0, keepdims=True) + 10**(-20))
        
    def ysigfunc(sigma, xdata, ydata):
        yfit = np.zeros((bins-bins_l_ext-bins_r_ext, num_multi_r_out))
        for i in range(bins-bins_l_ext-bins_r_ext):
            for j in range(num_multi_r_out):
                disp = np.exp(-0.5 * ((zilist - i) / abs(sigma[j])) ** 2)
                norm = np.sum(disp)
                yfit[:, j] = yfit[:, j] + xdata[i, j] * disp / norm
        yfit = np.clip(yfit, 10**(-5), 1-10**(-5))
        return -1 * np.sum(ydata * np.log(yfit))
        
    sigma0 = np.ones(num_multi_r_out)
    res = least_squares(ysigfunc, sigma0, bounds=(0, 1000), args=(xdata, ydata))
    sigma_disp = res.x
    print ('label_dispersion:', sigma_disp)

    
    ##### (b) Determining the scatterings for the means #####
    
    var_scatter_ini = (zphoto_train_sub - z_train_sub) ** 2
    z_select = []
    var_select = []
    
    for i in range(num_multi_r_out): 
        z_select.append([])
        var_select.append([])
        
    for j in range(rbins):
        index_r_out = min(max(0, int(j / (1 + rbins_compress))), num_multi_r_out - 1)
        z_select[index_r_out].append(z_train_sub[ri_train_sub == j])
        var_select[index_r_out].append(var_scatter_ini[ri_train_sub == j])
        if j > 0 and (j + 1) % (1 + rbins_compress) == 0:  #interlacing
            z_select[index_r_out+1].append(z_train_sub[ri_train_sub == j])
            var_select[index_r_out+1].append(var_scatter_ini[ri_train_sub == j])
            
    sigma_sq_scatter = np.zeros((num_multi_r_out, bins-bins_l_ext-bins_r_ext))
    
    for i in range(num_multi_r_out):
        zfit = np.concatenate(z_select[i], 0)
        vfit_ini = np.concatenate(var_select[i], 0)
        vfit = np.zeros(bins-bins_l_ext-bins_r_ext)
        for j in range(bins-bins_l_ext-bins_r_ext):
            zfilt = (zfit > wbin * j) & (zfit < wbin * (j+1))
            vfit[j] = 0.5 * (np.median(vfit_ini[zfilt]) + np.mean(vfit_ini[zfilt])) / wbin**2

        if len(vfit[vfit > 0]) < 1: continue
        vfit_pre = 0.0
        vfit_filt = (zilist < np.min(zilist[vfit > 0])) | (zilist > np.max(zilist[vfit > 0]))
        for j in range(bins-bins_l_ext-bins_r_ext):
            if j > np.max(zilist[vfit > 0]): continue
            if j < np.min(zilist[vfit > 0]): continue 
            exp_term = np.exp(-0.5 * ((zilist - j) / sigma_disp[i]) ** 2)
            exp_term[vfit_filt] = 0.0
            if vfit[j] > 0:
                sigma_sq_scatter[i] = sigma_sq_scatter[i] + vfit[j] * exp_term / np.sum(exp_term)
                vfit_pre = vfit[j]
            else:
                sigma_sq_scatter[i] = sigma_sq_scatter[i] + vfit_pre * exp_term / np.sum(exp_term)
    sigma_scatter = np.sqrt(sigma_sq_scatter + np.expand_dims(sigma_disp, 1) ** 2)
    print ('sigma_scatter_max:', np.max(sigma_scatter, 1))
        
    

###################################
##### Network & Cost Function #####
###################################
   
  
##### Placeholders #####
   
   
lr = tf.placeholder(tf.float32, shape=[], name='lr')   
x = tf.placeholder(tf.float32, shape=[None, imag_size, imag_size, channels], name='x')
reddening = tf.placeholder(tf.float32, shape=[None], name='reddening')
if ydisp_shift and multi_r_out:
    y = tf.placeholder(tf.float32, shape=[None, rbins+num_multi_r_out*bins], name='y')
elif multi_r_out:
    y = tf.placeholder(tf.float32, shape=[None, rbins+bins], name='y')
else:
    y = tf.placeholder(tf.float32, shape=[None, bins], name='y')


##### Network #####
   
    
zlogits = z_estimator(input=x, net=net, reddening=reddening, bins=bins, multi_r_out=multi_r_out, num_multi_r_out=num_multi_r_out, name='netJ_orig')
    

##### Output & cost function #####


if multi_r_out:
    yr = y[:, :rbins]
    yz = y[:, rbins:]
else:
    yz = y
                       

if multi_r_out:
    rbins_compress = int((rbins - 1) / num_multi_r_out)  #1  #2
    yr_compress = []
    for i in range(num_multi_r_out):
        yr_compress.append(tf.reduce_sum(yr[:, max(0,(rbins_compress+1)*i-1):(rbins_compress+1)*(i+1)], 1))
    yr_compress = tf.stack(yr_compress, 1)
                
    p_out = []
    cost_z = 0
    for i in range(num_multi_r_out):
        if ydisp_shift: yz_i = yz[:, i*bins:(i+1)*bins]
        else: yz_i = yr_compress[:, i:i+1] * yz + (1 - yr_compress[:, i:i+1]) * tf.ones_like(yz) / bins
        cost_z = cost_z + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yz_i, logits=zlogits[:, i*bins:(i+1)*bins]))
        p_out.append(tf.nn.softmax(zlogits[:, i*bins:(i+1)*bins]))
                
    yr_compress = yr_compress / tf.reduce_sum(yr_compress, 1, keepdims=True)
    cost_z = cost_z + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yr_compress, logits=zlogits[:, -1*num_multi_r_out:]))
    p_out = tf.reduce_sum(tf.stack(p_out, 2) * tf.expand_dims(tf.nn.softmax(zlogits[:, -1*num_multi_r_out:]), 1), 2)

else:
    p_out = tf.nn.softmax(zlogits)        
    cost_z = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yz, logits=zlogits))

        
        
#############################
##### Training & Saving #####
#############################
    
    
##### Select mini-batches #####

    
def img_reshape(a):
    mode = np.random.random()
    if mode < 0.25: a = np.rot90(a, 1)
    elif mode < 0.50: a = np.rot90(a, 2)
    elif mode < 0.75: a = np.rot90(a, 3)
    else: pass
        
    mode = np.random.random()
    if mode < 1 / 3.0: a = np.flip(a, 1)
    elif mode < 2 / 3.0: a = np.flip(a, 0)
    else: pass            
    return a



def get_next_batch(batch):
    index = np.random.choice(len(obj_train), batch)
    obj = obj_train[index]
    z = z_train[index]
    reddening = ebv_train[index]
    r = r_train[index]
    
    x = np.zeros((batch, imag_size, imag_size, channels))

    if ydisp_shift:
        yz = np.ones((batch, bins * num_multi_r_out)) / bins
    else:
        yz = np.zeros((batch, bins))
        
    if multi_r_out:
        yr = np.zeros((batch, rbins))
               
    for i in range(batch):
        if multi_r_out:
            index_r_in = max(0, min(rbins - 1, int((r[i] - r_min) / rwbin)))
            yr[i, index_r_in] = 1.0
            
        if use_SDSS:
            img = np.load(directory_SDSSdata + 'data/' + str(obj[i]) + '.py.npy')
        elif use_CFHTmix:
            img = np.array(list(imgs_ini[obj[i]]))
        img = img_reshape(img)        
        x[i] = img_norm(img, NormMethod)
        
        z_index = max(0, min(bins - 1, int((z[i] - z_min) / wbin))) + bins_l_ext
             
        if ydisp_shift:
            zilist_ext = np.arange(bins)
            index_r_out = min(max(0, int(index_r_in / (1 + rbins_compress))), num_multi_r_out - 1)
    
            filt_scatter = np.concatenate([np.zeros(bins_l_ext), dist2d_zr_train_norm[:, index_r_in], np.zeros(bins_r_ext)], 0)
            sigma_scatter_index_r_out = sigma_scatter[index_r_out][z_index-bins_l_ext]
            sigma_disp_index_r_out = sigma_disp[index_r_out]
                    
            if sigma_scatter_index_r_out < 0.1:
                p_scatter = np.zeros(bins)
                p_scatter[int(z_index)] = 1.0
            else:
                p_scatter = np.exp(-0.5 * ((zilist_ext - z_index) / sigma_scatter_index_r_out) ** 2)
                
            if shiftlabel == 0:
                z_index_ad = z_index
            else:
                p_scatter = p_scatter * filt_scatter
                p_scatter = p_scatter / np.sum(p_scatter)
                z_cm = np.sum(zilist_ext * p_scatter)
                z_index_ad = 2 * z_index - z_cm
                    
            if sigma_disp_index_r_out < 0.1 or softlabel == 0:
                disp = np.zeros(bins)
                disp[min(bins-1, max(int(z_index_ad), 0))] = 1.0
            else:
                disp = np.exp(-0.5 * ((zilist_ext - z_index_ad) / sigma_disp_index_r_out) ** 2)
            yz[i][index_r_out*bins: (index_r_out+1)*bins] = disp / np.sum(disp)


            if index_r_in > 0 and (index_r_in + 1) % (1 + rbins_compress) == 0: #interlacing
                sigma_scatter_index_r_out = sigma_scatter[index_r_out+1][z_index-bins_l_ext]
                sigma_disp_index_r_out = sigma_disp[index_r_out+1]
                    
                if sigma_scatter_index_r_out < 0.1:
                    p_scatter = np.zeros(bins)
                    p_scatter[int(z_index)] = 1.0
                else:
                    p_scatter = np.exp(-0.5 * ((zilist_ext - z_index) / sigma_scatter_index_r_out) ** 2)
                    
                if shiftlabel == 0:
                    z_index_ad = z_index
                else:
                    p_scatter = p_scatter * filt_scatter
                    p_scatter = p_scatter / np.sum(p_scatter)
                    z_cm = np.sum(zilist_ext * p_scatter)
                    z_index_ad = 2 * z_index - z_cm
                        
                if sigma_disp_index_r_out < 0.1 or softlabel == 0:
                    disp = np.zeros(bins)
                    disp[min(bins-1, max(int(z_index_ad), 0))] = 1.0
                else:
                    disp = np.exp(-0.5 * ((zilist_ext - z_index_ad) / sigma_disp_index_r_out) ** 2)
                yz[i][(index_r_out+1)*bins: (index_r_out+2)*bins] = disp / np.sum(disp)

        else:
            yz[i, z_index] = 1.0 
    
    if multi_r_out:
        y = np.concatenate([yr, yz], 1)
    else: y = yz
        
    return x, y, reddening

      

##### Get test results #####


def get_cost_z_stats(session, imgs_q, ebv_q, z_q, y_q):
    batch = 128
    p_out_q = np.zeros(y_q[:, -1*bins:].shape)
    cost_z_q = 0

    for i in range(0, len(z_q), batch):
        index_i = np.arange(i, min(i + batch, len(z_q)))
        imgs_batch = imgs_q[index_i]                
        y_batch = y_q[index_i]
        ebv_batch = ebv_q[index_i]
                        
        feed_dict = {x:imgs_batch, y:y_batch, reddening:ebv_batch}
        p_out_q_i, cost_z_q_i = session.run([p_out, cost_z], feed_dict = feed_dict)
        p_out_q[index_i] = p_out_q_i
        cost_z_q = cost_z_q + cost_z_q_i * len(imgs_batch)
    cost_z_q = cost_z_q / len(z_q)

    zlist = (0.5 + np.arange(bins)) * wbin
    zphot_q = np.zeros(len(z_q))
    for i in range(len(z_q)):
        zphot_q[i] = zlist[np.argmax(p_out_q[i])]
    zphot_q = zphot_q - zl_ext
        
    deltaz = (zphot_q - z_q) / (1 + z_q)
    residual = np.mean(deltaz)
    sigma_mad = 1.4826 * np.median(abs(deltaz - np.median(deltaz)))    
    if use_SDSS: eta_th = 0.05
    elif use_CFHTmix: eta_th = 0.15
    eta = len(deltaz[abs(deltaz) > eta_th]) / float(len(deltaz))
    crps = np.mean(np.sum((np.cumsum(p_out_q, 1) - np.cumsum(y_q[:, -1*bins:], 1)) ** 2, 1)) * wbin
        
    if test_phase: bins_wide = 40
    else: bins_wide = 8
    residual_list = np.zeros(bins_wide)
    for i in range(bins_wide):
        residual_list[i] = np.mean(deltaz[(z_q >= i * z_max / bins_wide) & (z_q < (i + 1) * z_max / bins_wide)])
        
    print ('num:', len(z_q))
    print ('cost:', cost_z_q)
    print ('residual_list:', residual_list)
    print ('residual, sigmad, eta, crps:', residual, sigma_mad, eta, crps)
    return cost_z_q, p_out_q, residual, sigma_mad, eta, crps



##### Session, saver or optimizer #####


if use_cpu: session_conf = tf.ConfigProto(device_count={'GPU':0})#log_device_placement=True
else:
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_conf)


if not test_phase:
    tvars = tf.trainable_variables()
    if parttrain:
        tvars = [var for var in tvars if ('fc2' in var.name)]            
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    if clip_grad:
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
    optimizer = optimizer.minimize(cost_z, var_list=tvars)        
    session.run(tf.global_variables_initializer())
        
                
if test_phase or pretrain:   
    if alter_checkpoint and ((tstep < 2 and test_phase) or (tstep > 1 and not test_phase)):
        f = open(model_load + 'checkpoint', 'r')
        ss = f.readlines()
        f = open(model_load + 'checkpoint', 'w')
        f.write('model_checkpoint_path: "-' + str(ite_altercheckpoint-1) + '"\n' + 'all_model_checkpoint_paths: "-' + str(ite_altercheckpoint-1) + '"')
        f.close()
    tvars = tf.trainable_variables()
    if tstep == 3:
        tvars = [var for var in tvars if ('fc2' not in var.name)]
    saver = tf.train.Saver(var_list=tvars)
    saver.restore(session, tf.train.latest_checkpoint(model_load))
    if alter_checkpoint and ((tstep < 2 and test_phase) or (tstep > 1 and not test_phase)):
        f = open(model_load + 'checkpoint', 'w')
        f.write(ss[0]+ss[1])
        f.close()
        
if tstep == 2 and flat_threshold == 10000:
    tvars = tf.trainable_variables()
    saver = tf.train.Saver()
    saver.save(session, model_savepath, 0)                


##### Training #####


def Train(i, th):
    global x_, y_, reddening_, running
    if th == 0:
        feed_dict = {x:x_, y:y_, reddening:reddening_, lr:learning_rate} 
        
        if i == 0 or (i + 1) % ite_test_print == 0:
            print ('iteration:', i+1, 'learning_rate:', learning_rate, 'batch:', batch_train, 'time:', str((time.time() - start) / 60) + ' minutes')
            print ('cost_training:', session.run(cost_z, feed_dict = feed_dict))
            print ('outputs_test:')
            outputs_test = get_cost_z_stats(session, imgs_test, ebv_test, z_test, y_test)
        for t in range(repetition_per_ite):
            session.run(optimizer, feed_dict = feed_dict)
        running = 0
        
    else:
        def read_data(index_j):
            x_j, y_j, reddening_j = get_next_batch(len(index_j))
            while True:
                if running == 0: break
            x_[index_j] = x_j
            y_[index_j] = y_j
            reddening_[index_j] = reddening_j

        for j in range(1, num_threads):
            if th == j:
                index_j = np.arange((j-1)*batch_th, max(batch_train-1, j*batch_th))
                if len(index_j) == 0: continue
                read_data(index_j)
                   


if not test_phase and not (tstep == 2 and flat_threshold == 10000):
    x_, y_, reddening_ = get_next_batch(batch_train)
    
    learning_rate = learning_rate_ini
    start = time.time()
    print ('Start training...')
       
    for i in range(iterations):
        if (i + 1) % learning_rate_step == 0: learning_rate = learning_rate / lr_reduce_factor
        running = 1
        
        threads = []
        for th in range(num_threads):
            t = threading.Thread(target = Train, args = (i, th))
            threads.append(t)
        for th in range(num_threads):
            threads[th].start()
        for th in range(num_threads):
            threads[th].join()    
        
        if (i + 1) in ite_point_save:
            saver = tf.train.Saver()
            saver.save(session, model_savepath, i)
            

##### Saving #####


if test_phase: 
    cost_test, probas_test, bias_test, sigma_mad_test, eta_test, crps_test = get_cost_z_stats(session, imgs_test, ebv_test, z_test, y_test)
           
    if save_train:
        cost_train, probas_train, bias_train, sigma_mad_train, eta_train, crps_train = get_cost_z_stats(session, imgs_train, ebv_train, z_train, y_train)
        np.savez(z_savepath, z_train=z_train, y_train=y_train, ebv_train=ebv_train, zphot_train=zphot_train, 
                     z_test=z_test, y_test=y_test, ebv_test=ebv_test, zphot_test=zphot_test,
                                  r_train=r_train, rerr_train=rerr_train, r_test=r_test, rerr_test=rerr_test,
                                  i_train=i_train, ierr_train=ierr_train, i_test=i_test, ierr_test=ierr_test,
                                  cost_train=cost_train, cost_test=cost_test,
                                  probas_train=probas_train, bias_train=bias_train, sigma_mad_train=sigma_mad_train, eta_train=eta_train, crps_train=crps_train,
                                  probas_test=probas_test, bias_test=bias_test, sigma_mad_test=sigma_mad_test, eta_test=eta_test, crps_test=crps_test)
    else: 
        np.savez(z_savepath, z_test=z_test, y_test=y_test, ebv_test=ebv_test, zphot_test=zphot_test,
                                  r_test=r_test, rerr_test=rerr_test,
                                  i_test=i_test, ierr_test=ierr_test,
                                  cost_test=cost_test, 
                                  probas_test=probas_test, bias_test=bias_test, sigma_mad_test=sigma_mad_test, eta_test=eta_test, crps_test=crps_test)
print (fx)
