import numpy as np



##### Get the variances for photo-z estimates (as a function of spec-z)
#####
##### Inputs:
##### zspec --- an array of spec-z (instance-wise)
##### zphoto --- an array of photo-z (instance-wise)
##### r --- an array of r-band magnitude (instance-wise)
##### siglist --- an array of fitted per-magnitude-output-bin labeling dispersions
##### zmax --- the upper boundary of redshift
##### zbins --- the number of redshift bins per output
##### rmin --- the lower boundary of magnitude
##### r_outs --- the number of magnitude output bins 
#####  
##### Outputs[0] --- an array of variances (instance-wise)
##### Outputs[1] --- a grid of variances (bin-wise)

def get_variances(zspec, zphoto, r, siglist, zmax=0.4, zbins=180, rmin=12.5, r_outs=6):
    wbin = zmax / zbins
    r_rows = 2 * r_outs - 1
    ri = np.arange(r_rows + 3) * 0.5 + rmin - 0.5
    ri[0] = 0.0
    ri[-1] = 100.0

    zvar_out = np.zeros(zspec.shape)
    zvar_grid = np.zeros((r_rows, zbins))
    zvar_ini = (zspec - zphoto) ** 2
    zi = np.arange(zbins)
    
    for k in range(r_outs):
        rfilt = (r > ri[2*k]) & (r <= ri[2*k+3])
        if len(zspec[rfilt]) == 0: continue

        vfit = np.zeros(zbins)
        for i in range(zbins):
            zfilt = (zspec[rfilt] > wbin * i) & (zspec[rfilt] < wbin * (i+1))
            vfit[i] = 0.5 * (np.median(zvar_ini[rfilt][zfilt]) + np.mean(zvar_ini[rfilt][zfilt])) / wbin**2
        
        vfit_filt = (zi < np.min(zi[vfit > 0])) | (zi > np.max(zi[vfit > 0]))
        vfit_sm = 0
        vfit_pre = 0
        for i in range(zbins):
            if i > np.max(zi[vfit > 0]): continue
            if i < np.min(zi[vfit > 0]): continue
            exp_term = np.exp(-0.5 * ((zi - i) / siglist[k])**2)
            exp_term[vfit_filt] = 0.0
            if vfit[i] > 0:
                vfit_sm = vfit_sm + vfit[i] * exp_term / np.sum(exp_term)
                vfit_pre = vfit[i]
            else: 
                vfit_sm = vfit_sm + vfit_pre * exp_term / np.sum(exp_term)

        index_k = np.arange(len(zspec))[rfilt]
        for j in range(len(zspec[rfilt])):
            z_index = max(0, min(zbins - 1, int((zspec[rfilt][j] - 0.0) / wbin)))
            if zvar_out[rfilt][j] > 0: zvar_out[index_k[j]] = 0.5 * (zvar_out[rfilt][j] + vfit_sm[z_index])
            else: zvar_out[index_k[j]] = vfit_sm[z_index]
        
        zvar_grid[2*k] = vfit_sm
        if k < r_outs - 1: zvar_grid[2*k+1] = vfit_sm
        if k > 0: zvar_grid[2*k-1] = 0.5 * (zvar_grid[2*k-1] + vfit_sm)
            
        print ('r_multiout:', k+1)            
    return zvar_out * wbin**2, zvar_grid * wbin**2



##### Resample the training set for Step 4, in order to match p(zphoto_test) and p(zspec_train), 
##### where p(zphoto_test) is regarded as the expected p(zphoto_train).
#####
##### Inputs:
##### zphoto_test --- an array of photo-z from the test set (instance-wise)
##### r_test --- an array of r-band magnitude from the test set (instance-wise)    
##### zspec_train --- an array of spec-z from the training set (instance-wise)
##### zphoto_train --- an array of photo-z from the training set (instance-wise)
##### r_train --- an array of r-band magnitude from the training set (instance-wise)
##### size_resample --- the size of the resampled training set
##### zmax --- the upper boundary of redshift
##### zbins --- the number of redshift bins per output
##### rmin --- the lower boundary of magnitude
##### r_outs --- the number of magnitude output bins
#####  
##### Outputs[0] --- an array of spec-z from the resampled training set (instance-wise)
##### Outputs[1] --- an array of photo-z from the resampled training set (instance-wise)
##### Outputs[2] --- an array of r-band magnitude from the resampled training set (instance-wise)

def trainset_resample(zphoto_test, r_test, zspec_train, zphoto_train, r_train, size_resample, zmax=0.4, zbins=180, rmin=12.5, r_outs=6):
    wbin = zmax / zbins
    r_rows = 2 * r_outs - 1    
    ri = np.arange(r_rows + 3) * 0.5 + rmin - 0.5
    ri[0] = 0.0
    ri[-1] = 100.0

    rescale = size_resample / float(len(zphoto_test))
    index_list = np.arange(len(zspec_train))
    index_select = []
    
    for j in range(r_rows):
        zphoto_test_j = zphoto_test[(r_test > ri[j]) & (r_test <= ri[j+1])]
        if len(zphoto_test_j) == 0: continue
    
        index_j = index_list[(r_train > ri[j]) & (r_train <= ri[j+1])]
        dist_j = np.histogram(zspec_train[index_j], zbins, (0, zmax))[0]
        zlist_j = np.arange(zbins)[dist_j > 0]
        for i in range(zbins):
            zphoto_test_ij = zphoto_test_j[(zphoto_test_j > i * wbin) & (zphoto_test_j <= (1+i) * wbin)]
            if len(zphoto_test_ij) == 0: continue
            num_i = rescale * len(zphoto_test_ij)
            num_i = int(num_i) + int(2 * (num_i - int(num_i)))
            
            i2 = zlist_j[np.argmin(abs(zlist_j - i))]
            index_ij = index_j[(zspec_train[index_j] > i2 * wbin) & (zspec_train[index_j] <= (1+i2) * wbin)]
            index_random = np.concatenate([np.random.choice(index_ij, len(index_ij), replace=False), np.random.choice(index_ij, num_i, replace=True)])
            index_select.append(index_random[:num_i])
    
    index_select = np.concatenate(index_select)
    return zspec_train[index_select], zphoto_train[index_select], r_train[index_select]




##### Step 4: correct photo-z-dependent residuals using the resampled training set
##### (after applying "trainset_resample" such that p(zspec_train) and p(zphoto_test) are matched).
#####
##### Inputs:
##### zphoto_test --- an array of photo-z from the test set (instance-wise)
##### r_test --- an array of r-band magnitude from the test set (instance-wise)    
##### zspec_train --- an array of spec-z from the resampled training set (instance-wise)
##### zphoto_train --- an array of photo-z from the resampled training set (instance-wise)
##### r_train --- an array of r-band magnitude from the resampled training set (instance-wise)
##### zmax --- the upper boundary of redshift
##### zbins_wide --- the number of wide redshift bins for computing the corrections
##### rmin --- the lower boundary of magnitude
##### r_outs --- the number of magnitude output bins
#####  
##### Output --- an array of corrected photo-z from the test set (instance-wise)

def residuals_zphoto_corr(zphoto_test, r_test, zspec_train, zphoto_train, r_train, zmax=0.4, zbins_wide=40, rmin=12.5, r_outs=6):
    r_rows = 2 * r_outs - 1    
    ri = np.arange(r_rows + 3) * 0.5 + rmin - 0.5
    ri[0] = 0.0
    ri[-1] = 100.0
    zphoto_corr = np.array(list(zphoto_test))

    for i in range(zbins_wide):
        for j in range(r_rows):
            filt_test = (zphoto_test > i * zmax / zbins_wide) & (zphoto_test < (i+1) * zmax / zbins_wide) & (r_test > ri[j]) & (r_test < ri[j+1])
            filt_train = (zphoto_train > i * zmax / zbins_wide) & (zphoto_train < (i+1) * zmax / zbins_wide) & (r_train > ri[j]) & (r_train < ri[j+1])
            if len(zphoto_test[filt_test]) == 0 or len(zphoto_train[filt_train]) == 0:
                continue    
            zphoto_corr[filt_test] = zphoto_test[filt_test] - np.mean((zphoto_train - zspec_train)[filt_train])
    return zphoto_corr



##### Estimate sigma_MAD as a function of r-band magnitude. "get_sigmad" makes use of "sigmad_permag" that gives per-magnitude estimates.
#####
##### Inputs for "get_sigmad":
##### zspec --- an array of spec-z (instance-wise)
##### zphoto --- an array of photo-z (instance-wise)
##### r --- an array of r-band magnitude (instance-wise)
##### rmin --- the lower boundary of magnitude
##### r_outs --- the number of magnitude output bins
##### 
##### Outputs[0, 1] --- an array of center magnitude values (and uncertainties) for plotting
##### Outputs[2, 3] --- an array of estimated sigma_MAD (and uncertainties)

def sigmad_permag(zspec_permag, zphoto_permag, r_permag):
    zdelta = (zphoto_permag - zspec_permag) / (1 + zspec_permag)
    sigmad_list = np.zeros(100)
    rcenter_list = np.zeros(100)
    for i in range(100):  #bootstrap
        select = np.random.choice(len(zspec_permag), len(zspec_permag), replace=True)
        sigmad_list[i] = 1.4826 * np.median(abs(zdelta[select] - np.median(zdelta[select])))
        rcenter_list[i] = np.mean(r_permag[select])
    return np.mean(rcenter_list), np.sqrt(np.sum((rcenter_list - np.mean(rcenter_list)) ** 2) / (100 - 1)), np.mean(sigmad_list), np.sqrt(np.sum((sigmad_list - np.mean(sigmad_list)) ** 2) / (100 - 1))

def get_sigmad(zspec, zphoto, r, rmin=12.5, r_outs=6): 
    r_rows = 2 * r_outs - 1       
    ri = np.arange(r_rows + 1) * 0.5 + rmin
    ri[0] = 0.0
    ri[-1] = 100.0
    
    sigmad = np.zeros(r_rows)
    err_sigmad = np.zeros(r_rows)
    rlist = np.zeros(r_rows)
    err_rlist = np.zeros(r_rows)
    for i in range(r_rows):
        filt = (r > ri[i]) & (r <= ri[i+1])
        rlist[i], err_rlist[i], sigmad[i], err_sigmad[i] = sigmad_permag(zspec[filt], zphoto[filt], r[filt])
    return rlist, err_rlist, sigmad, err_sigmad



##### Estimate the mean residuals as a function of spec-z using pre-estimated variances
#####
##### Inputs:
##### zspec --- an array of spec-z (instance-wise)
##### zphoto --- an array of photo-z (instance-wise)
##### zvar --- an array of pre-estimated variances (instance-wise)
##### zmax --- the upper boundary of redshift
##### zbins_wide --- the number of wide redshift bins for plotting
##### 
##### Outputs[0] --- an array of spec-z bins for plotting
##### Outputs[1] --- an array of residuals (bin-wise)
##### Outputs[2] --- an array of residual uncertainties (bin-wise)

def get_residuals_zspec(zspec, zphoto, zvar, zmax=0.4, zbins_wide=40):
    zvar = np.clip(zvar, 10**(-5), 10**10)
    zdelta = (zphoto - zspec) / (1 + zspec)
    zvar_zdelta = zvar / (1 + zspec)**2
    zresidual = np.zeros(zbins_wide)
    err_zresidual = np.zeros(zbins_wide)

    for i in range(zbins_wide):
        filt = (zspec > i * zmax / zbins_wide) & (zspec < (i+1) * zmax / zbins_wide)
        if len(zdelta[filt]) == 0: continue
        zresidual[i] = np.sum(zdelta[filt] / zvar_zdelta[filt]) / np.sum(1 / zvar_zdelta[filt])
        err_zresidual[i] = np.sqrt(1 / np.sum(1 / zvar_zdelta[filt]))
        
    zlist = ((np.arange(zbins_wide) + 0.5) * zmax / zbins_wide)[err_zresidual > 0]
    zresidual = zresidual[err_zresidual > 0]
    err_zresidual = err_zresidual[err_zresidual > 0]
    return zlist, zresidual, err_zresidual



##### Estimate the mean residuals as a function of photo-z, whose RMSs are taken as uncertainties.
#####
##### Inputs:
##### zspec --- an array of spec-z (instance-wise)
##### zphoto --- an array of photo-z (instance-wise)
##### zmax --- the upper boundary of redshift
##### zbins_wide --- the number of wide redshift bins for plotting
##### 
##### Outputs[0] --- an array of photo-z bins for plotting
##### Outputs[1] --- an array of residuals (bin-wise)
##### Outputs[2] --- an array of residual uncertainties (bin-wise)

def get_residuals_zphoto(zspec, zphoto, zmax=0.4, zbins_wide=40):
    zdelta = (zphoto - zspec) / (1 + zspec)
    zresidual = np.zeros(zbins_wide)
    err_zresidual = np.zeros(zbins_wide)

    for i in range(zbins_wide):
        filt = (zphoto > i * zmax / zbins_wide) & (zphoto < (i+1) * zmax / zbins_wide)
        if len(zdelta[filt]) == 0: continue
        zresidual[i] = np.mean(zdelta[filt])
        err_zresidual[i] = np.std(zdelta[filt]) / np.sqrt(len(zdelta[filt]) - 1)

    zlist = ((np.arange(zbins_wide) + 0.5) * zmax / zbins_wide)[err_zresidual > 0]
    zresidual = zresidual[err_zresidual > 0]
    err_zresidual = err_zresidual[err_zresidual > 0]
    return zlist, zresidual, err_zresidual



##### Linear fit of the mean residuals as a function of spec-z or photo-z.
##### Outputs --- fitted parameters and uncertainties

from scipy.optimize import curve_fit

def func_linear(z, a, b):
    return a * z + b

def fit_z_res(z, residuals, err_residuals):
    popt, pcov = curve_fit(func_linear, z, residuals, sigma=err_residuals)
    return popt[0], popt[1], pcov[0][0]**0.5, pcov[1][1]**0.5



##### Estimate the Total Variation distance between p(zphoto) and p(zspec).
#####
##### Inputs:
##### zspec --- an array of spec-z (instance-wise)
##### zphoto --- an array of photo-z (instance-wise)
##### zlist --- an array of photo-z bins from "get_residuals_zspec"
##### zresidual --- an array of residuals from "get_residuals_zspec"
##### err_zresidual --- an array of residual uncertainties from "get_residuals_zspec"
##### zmax --- the upper boundary of redshift
##### zbins --- the number of redshift bins per output
##### zbins_wide --- the number of wide redshift bins
##### 
##### Outputs[0, 1] --- the Total Variation distance with the actual distribution and its uncertainty
##### Outputs[2, 3] --- the Total Variation distance with the simulated distribution and its uncertainty (expected to have no mode collapse)

def get_total_variation(zspec, zphoto, zlist, zresidual, err_zresidual, zmax=0.4, zbins=180, zbins_wide=40):
    tvlist = np.zeros(100)
    for i in range(100):  #bootstrap
        select = np.random.choice(len(zspec), len(zspec), replace=True)
        dspec = np.histogram(zspec[select], zbins, (0, zmax))[0] / float(len(zspec))
        dphoto = np.histogram(zphoto[select], zbins, (0, zmax))[0] / float(len(zspec))
        tvlist[i] = np.sum(0.5 * abs(dphoto - dspec))
    
    zbins_wide = 40
    zmean = zresidual * (1 + zlist) + zlist
    zdisp = err_zresidual * (1 + zlist)
    tvlist_sm = np.zeros(100)
    for i in range(100):  #bootstrap
        zphoto_select = []
        for j in range(zbins_wide):
            filt = (zspec > j * zmax / zbins_wide) & (zspec <= (j+1) * zbins_wide / zbins_wide)
            if (j+0.5) * zmax / zbins_wide in zlist: zphoto_select.append(np.random.normal(zmean[j], np.sqrt(len(zspec[filt])) * zdisp[j], len(zspec[filt])))
            else: zphoto_select.append(zspec[filt])
        zphoto_select = np.concatenate(zphoto_select)
        dspec = np.histogram(zspec, zbins, (0, zmax))[0] / float(len(zspec))
        dphoto = np.histogram(zphoto_select, zbins, (0, zmax))[0] / float(len(zspec))
        tvlist_sm[i] = np.sum(0.5 * abs(dphoto - dspec))
    return np.mean(tvlist), np.std(tvlist) * (100 / 99.0)**0.5, np.mean(tvlist_sm), np.std(tvlist_sm) * (100 / 99.0)**0.5



##### Estimate the Wasserstein distance between p(zphoto) and p(zspec) as a function of r-band magnitude.
#####
##### Inputs:
##### zspec --- an array of spec-z (instance-wise)
##### zphoto --- an array of photo-z (instance-wise)
##### r --- an array of r-band magnitude (instance-wise)
##### zmax --- the upper boundary of redshift
##### zbins --- the number of redshift bins per output
##### rmin --- the lower boundary of magnitude
##### rmax --- the upper boundary of magnitude
##### r_outs --- the number of magnitude bins
##### 
##### Outputs[0, 1] --- an array of center magnitude values (and uncertainties) for plotting
##### Outputs[2, 3] --- an array of estimated Wasserstein distance (and uncertainties)
    
def get_wasserstein_distance(zspec, zphoto, r, zmax=0.4, zbins=180, rmin=12.5, rmax=18.0, r_outs=6):
    ri = np.arange(r_outs + 1) * (rmax - rmin) / r_outs + rmin
    ri[0] = 0.0
    ri[-1] = 100.0    
    rcenter = np.zeros(r_outs)
    err_rcenter = np.zeros(r_outs)
    wdist = np.zeros(r_outs)
    err_wdist = np.zeros(r_outs)
    for j in range(r_outs):
        filt = (r > ri[j]) & (r <= ri[j+1])
        if len(zphoto[filt]) == 0: continue
        rcenter_list = np.zeros(100)
        wdist_list = np.zeros(100)
        for i in range(100):  #bootstrap
            select = np.random.choice(len(zspec[filt]), len(zspec[filt]), replace=True)            
            rcenter_list[j] = np.mean(r[filt][select])
            dspec = np.histogram(zspec[filt][select], zbins, (0, zmax))[0] / float(len(zspec[filt]))
            dphoto = np.histogram(zphoto[filt][select], zbins, (0, zmax))[0] / float(len(zspec[filt]))
            wdist_list[i] = np.sum(abs(np.cumsum(dphoto) - np.cumsum(dspec))) * zmax / zbins
        rcenter[j] = np.mean(rcenter_list)
        err_rcenter[j] = np.std(rcenter_list) * (100 / 99.0)**0.5
        wdist[j] = np.mean(wdist_list)
        err_wdist[j] = np.std(wdist_list) * (100 / 99.0)**0.5
    return rcenter, err_rcenter, wdist, err_wdist



##### Get the simulated residual curve (as a function of spec-z and marginalized over r-band magnitude).
##### "get_residuals_sim" makes use of "get_residuals_sim_permag" that makes per-magnitude simulations.
#####
##### Inputs for "get_residuals_sim":
##### zspec --- an array of spec-z (instance-wise)
##### r --- an array of r-band magnitude (instance-wise)
##### zvar_grid --- a grid of variances (bin-wise) from "get_variances"
##### k --- the factor for rescaling the variances
##### zmax --- the upper boundary of redshift
##### zbins --- the number of redshift bins per output
##### zbins_wide --- the number of wide redshift bins
##### rmin --- the lower boundary of magnitude
##### r_outs --- the number of magnitude output bins
#####
##### Output --- the simulated residual curve

def get_residuals_sim_permag(zspec_permag, zvar_permag, zmax=0.4, zbins=180, zbins_wide=40):
    zlist = (np.arange(zbins_wide) + 0.5) * zmax / zbins_wide
    dspec = np.histogram(zspec_permag, zbins_wide, (0, zmax))[0]
    residuals_sim = np.zeros(zbins_wide)
    zvar_rebin = np.mean(np.reshape(np.reshape(np.stack([zvar_permag, zvar_permag], 1), 2 * zbins), [zbins_wide, int(2*zbins/zbins_wide)]), 1)
    for i in range(zbins_wide):
        exp_term = np.exp(-0.5 * (zlist[i] - zlist) ** 2 / zvar_rebin[i]) * dspec
        if np.sum(exp_term) > 0: residuals_sim[i] = (np.sum(exp_term * zlist) / np.sum(exp_term) - zlist[i]) / (1 + zlist[i])
        else: residuals_sim[i] = 0
    return residuals_sim, dspec / (zvar_rebin + 10**(-20))

def get_residuals_sim(zspec, r, zvar_grid, k=1.0, zmax=0.4, zbins=180, zbins_wide=40, rmin=12.5, r_outs=6): 
    r_rows = 2 * r_outs - 1       
    ri = np.arange(r_rows + 1) * 0.5 + rmin
    ri[0] = 0.0
    ri[-1] = 100.0
    residuals_sim_grid = np.zeros((r_rows, zbins_wide))
    weights_grid = np.zeros((r_rows, zbins_wide))
    for i in range(r_rows):
        filt = (r > ri[i]) & (r <= ri[i+1])
        if len(r[filt]) == 0: continue
        residuals_sim_grid[i], weights_grid[i] = get_residuals_sim_permag(zspec[filt], k * zvar_grid[i], zmax=zmax, zbins=zbins, zbins_wide=zbins_wide)        
    return np.sum(weights_grid * residuals_sim_grid, 0) / np.sum(weights_grid, 0)

