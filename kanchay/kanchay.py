########### updated 09/10/2021 ###########

#USEFUL MODULES FOR ROTATIONAL PERIOD IN PYTHON
from .kanchay_tools import sigma_clip, norma_one, concat_lk
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import trange
import numpy as np
import pandas as pd
from starspot import rotation_tools as rt
import starspot as ss
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
lk.log.setLevel("DEBUG")


#########################################################################################################
####################################### TESS light curve ################################################
#########################################################################################################

plotpar = {'axes.labelsize': 25,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'text.usetex': True}
plt.rcParams.update(plotpar)

class kan(object):
    def __init__(self):
        self.starid   = None
        self.mission  = None
        self.exptime  = None
        self.store    = None
        self.plot     = None
    
    @classmethod
    def search_lk(self, starid, mission, exptime=None, store=None, plot_f=None):
        #mission: ‘Kepler’, ‘K2’, or ‘TESS’. By default, all will be returned.
        #author: Official Kepler, K2, and TESS pipeline products have author names ‘Kepler’, ‘K2’, and ‘SPOC’.
        #by default, author = SPOC = TESS
        #if not author:
        #    author = 'SPOC'
        #else:
        #    author = author
        if not store:
            store = 'no'
        else:
            store = store
        if not plot_f:
            plot_f = 'png'
        else:
            plot_f = plot_f
    
        #selecting the mission
        #TESS
        if mission == 'TESS':
            #searching data with Lightkurve
            #exptime: ‘long’ selects 10-min and 30-min cadence products; ‘short’ 
            #selects 1-min and 2-min products; ‘fast’ selects 20-sec products.
            #by default 120 = 2 min
            if not exptime:
                exptime = 120
            else:
                exptime = exptime
            #searching TESS Lightcurve
            author = 'SPOC'
            search = lk.search_lightcurve(starid, exptime=exptime, mission=mission, author=author)
            print (search)
            
        elif mission == 'Kepler':
            #by default 60 = 1 min
            if not exptime:
                exptime = 60
            else:
                exptime = exptime
            #searching Kepler Lightcurve
            author = 'Kepler'
            search = lk.search_lightcurve(starid, exptime=exptime, mission=mission, author=author)
            print (search)
        
        #elif mission == 'K2':
            #in construction
        
        try:       
            #sigma clipping
            x, y, yerr, std = [], [], [], []
            xf, yf, yerrf   = [], [], []
            for i in trange(len(search)):
                try:
                    lc = search[i].download()
                    lc = lc.remove_nans()#.remove_outliers()
            
                    t, f, ferr, stdev = sigma_clip(lc.time.jd, lc.flux.value, lc.flux_err.value)
                    x.append(t)
                    y.append(f)
                    yerr.append(ferr)
                    std.append(stdev)
                    #Normalizing to the mean
                    xnu, ynu, yerrnu = norma_one(lc.time.jd, lc.flux.value, lc.flux_err.value)
                    xn, yn, yerrn    = norma_one(x[i], y[i], yerr[i])
                    #Plotting individual sector
                    plt.figure(figsize=(16, 5))
                    #####plt.plot(lc.time.jd, lc.flux.value, "C1.", ms=3, alpha=.5)
                    plt.plot(xnu, ynu, "C1.", ms=3, alpha=.5)   #uncorrected light
                    plt.plot(xn, yn, "C0.", ms=1, alpha=.9)
                    if mission == 'TESS':
                        plt.savefig('TIC'+search.target_name[i]+'_'+search.mission[i]+'.'+plot_f, bbox_inches='tight', dpi=200)
                    else:
                        plt.savefig(search.target_name[i]+'_'+search.mission[i]+'.'+plot_f, bbox_inches='tight', dpi=200)
                    #Appending normalized results
                    xf.append(xn)
                    yf.append(yn)
                    yerrf.append(yerrn)
                    if store == 'yes':
                        #converting to csv
                        data = {'time_n':xn, 'flux_n':yn, 'err_flux_n':yerrn, 
                                'time':t, 'flux':f, 'err_flux':ferr}
                        df = pd.DataFrame(data)
                        if mission == 'TESS':
                            df.to_csv('TIC'+search.target_name[i]+'_'+search.mission[i]+'.csv', index=False)
                        else:
                            df.to_csv(search.target_name[i]+'_'+search.mission[i]+'.csv', index=False)
                except:
                    pass

            #Plotting results with gaps
            plt.figure(figsize=(16, 6))
        
            gs = GridSpec(2, 1)
            gs.update(hspace=0.)
        
            ax0  = plt.subplot(gs[0])
            plt.title(starid+' '+author, fontsize=20)
            start_times, stop_times = [], []
            for i in range(len(search)):
                ax0.plot(lc.time.jd, lc.flux.value, "C1.", ms=1, alpha=.5)
                ax0.plot(x[i], y[i], "C0.", ms=1, alpha=.5)
                start_times.append(min(x[i]))
                stop_times.append(max(y[i]))   
            #print(lc.time.jd)
            gaps = start_times[1:]
            for i in range(len(gaps)):
                ax0.axvline(gaps[i], color="k")
            
            ax1 = plt.subplot(gs[1])
            for i in range(len(search)):
                ax1.plot(xf[i], yf[i], "C0.", ms=1, alpha=.5)
        
            for i in range(len(gaps)):
                ax1.axvline(gaps[i], color="k")
            if mission == 'TESS':
                plt.savefig('TIC'+search.target_name[0]+'_all_sectors.'+plot_f, bbox_inches='tight', dpi=200)
            else:
                plt.savefig(search.target_name[0]+'_all_sectors.'+plot_f, bbox_inches='tight', dpi=200)
            
            return xf, yf, yerrf
    
        except:
            pass
        
    #########################################################################################################
    ####################################### Gaussian Process ################################################
    #########################################################################################################
    @classmethod
    def gp_jao(self, x, y, yerr, period, Q):
        
        with pm.Model() as model:

            # The mean flux of the time series
            mean = pm.Normal("mean", mu=0.0, sigma=10.0)

            # A jitter term describing excess white noise
            log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sigma=2.0)

            # A term to describe the non-periodic variability
            sigma = pm.InverseGamma(
                "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
            )
            rho = pm.InverseGamma(
                "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
            )

            # The parameters of the RotationTerm kernel
            sigma_rot = pm.InverseGamma(
                "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
            )
            log_period = pm.Normal("log_period", mu=np.log(period), sigma=2.0)
            period = pm.Deterministic("period", tt.exp(log_period))
            log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
            log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=2.0)
            f = pm.Uniform("f", lower=0.1, upper=1.0)

            # Set up the Gaussian Process model
            kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=Q)
            kernel += terms.RotationTerm(
                sigma=sigma_rot,
                period=period,
                Q0=tt.exp(log_Q0),
                dQ=tt.exp(log_dQ),
                f=f,
            )
            gp = GaussianProcess(
                kernel,
                t=x,
                diag=yerr ** 2 + tt.exp(2 * log_jitter),
                mean=mean,
                quiet=True,
            )

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            gp.marginal("gp", observed=y)

            # Compute the mean model prediction for plotting purposes
            pm.Deterministic("pred", gp.predict(y))

            # Optimize to find the maximum a posteriori parameters
            map_soln = pmx.optimize()
            
            #prediction model
            #mup, varp = pmx.eval_in_model(
                #gp.predict(x, return_var=True), map_soln)
            
            
            #Plotting Results
            fig = plt.figure(figsize=(12,6))
            #plt.plot(x, y, "k", label="data")

            plt.scatter(x, y, label="data", c='#85C1E9')
            plt.plot(x, map_soln["pred"], color="black", label="model", linewidth=2.) #+ map_soln["mean"]
            #sd = np.sqrt(var1)
            #art = plt.fill_between(x, mu1 + map_soln["mean"] + 3*sd, mu1 + map_soln["mean"] - 3*sd, 
            #                   color="red", alpha=0.6, label='prediction')
            #art.set_edgecolor("none")
            plt.xlim(x.min(), x.max())
            plt.ylim(y.min(), y.max())
            plt.legend(loc='best', fontsize=10)
            plt.xlabel("time [days]")
            plt.ylabel("relative flux [ppm]")
        
        with model:
            trace = pmx.sample(
            tune=1000,
            draws=1000,
            start=map_soln,
            cores=8,
            chains=2,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=[10863087, 10863088],
            )

        #Saving results
        Period = np.median(trace.posterior["period"])
        lower  = np.percentile(trace.posterior["period"], 16)
        upper  = np.percentile(trace.posterior["period"], 84)
        errm   = Period - lower
        errp   = upper - Period
        
        plt.figure(figsize=(9,6))
        period_samples = np.asarray(trace.posterior["period"]).flatten()
        plt.hist(period_samples, 15, histtype="step", color="k", density=True)
        plt.yticks([])
        #plt.axvline(np.log(period), color="k", lw=12, alpha=0.3, label='LS')
        plt.axvline(Period, color="b", lw=12, alpha=0.3, label='GP')
        plt.axvline(Period - errm, ls="--", color="C1")
        plt.axvline(Period + errp, ls="--", color="C1")

        plt.legend(loc='best', fontsize=15)
        plt.xlabel("Rotation period [days]")
        _ = plt.ylabel("Posterior density")
        plt.tight_layout()
        plt.savefig('Gaussian.png', dpi=200)
            
        return Period, lower, upper
    

    #########################################################################################################
    ####################################### Rotation Estimation #############################################
    #########################################################################################################
    @classmethod
    def rotation_calc(self, xf, yf, yerrf, xmin=None, xmax=None, gp=None):
        #by default, gp=no
        if not gp:
            gp = 'no'
        else:
            gp = gp
        #by default. xmin=0.1
        if not xmin:
            xmin = 0.1
        else:
            xmin = xmin
        #by default. xmax=10
        if not xmax:
            xmax = 10.
        else:
            xmax = xmax
        
        #calculating rotation
        rotate      = ss.RotationModel(xf, yf, yerrf)
        ls_period   = rotate.ls_rotation()
        acf_period  = rotate.acf_rotation(interval='TESS', cutoff=1, window_length=999, polyorder=3)
        period_grid = np.linspace(xmin, xmax, 1000)
        pdm_period  = rotate.pdm_rotation(period_grid)
        fig         = rotate.big_plot(methods=['ls', 'acf', 'pdm'], method_xlim=(-1, 30));
        plt.savefig('big_plot.png', dpi=200)
        #print(rotate.Rvar)
        
        #bining light curve
        #240s = 0.002777778d
        #30m  = 0.020833333d
        lc = lk.LightCurve(time=xf, flux=yf, flux_err=yerrf)
        lcb = lc.bin(time_bin_size=0.020833333)
        lcb = lcb.remove_nans()
        #lcb.scatter()
        
        xfb    = lcb.time.jd
        yfb    = lcb.flux.value
        yerrfb = lcb.flux_err.value
        
        if gp == 'yes':
            Period, lower, upper = self.gp_jao(xfb, yfb, yerrfb, pdm_period[0],Q=1 / 3.0)
            GP_ma = upper - Period
            GP_me = Period - lower
            
            data = {'LS':ls_period, 'ACF':acf_period, 'PDM':[pdm_period[0]], 'err_PDM':[pdm_period[1]], 
                    'GP':[Period], '+GP':[GP_ma], '-GP':[GP_me]}
            df = pd.DataFrame(data)
            
            return df
            
        data = {'LS':ls_period, 'ACF':acf_period, 'PDM':[pdm_period[0]], 'err_PDM':[pdm_period[1]]}
        df = pd.DataFrame(data)
    
        return df

