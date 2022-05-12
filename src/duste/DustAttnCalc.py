""" DustAttnCalc
Given stellar masses, star formation rates, stellar metallicities, redshifts, axis ratios, calculate dust attenuation curves based on the hierarchical Bayesian model from Nagaraj+22a (in prep)
Author: Gautam Nagaraj--gxn75@psu.edu
"""

__all__ = ["regular_grid_interp_scipy","mass_completeness","get_dust_attn_curve_d2","get_dust_attn_curve_d1","getMargSample","getTraceInfo","getModelSamplesI","plotDustAttn","plotDust12","DustAttnCalc"]

import numpy as np 
import os.path as op
from sedpy.attenuation import noll
from scipy.stats import truncnorm
import arviz as az
import argparse as ap
from glob import glob
import fnmatch
from astropy.table import Table
from distutils.dir_util import mkpath
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGIScipy
from dynesty.utils import resample_equal
import pkg_resources
import seaborn as sns
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

def parse_args(argv=None):
    """ Tool to parse arguments from the command line. The entries should be self-explanatory. They include all of the options in the DustAttnCalc class. """
    parser = ap.ArgumentParser(description="DustAttnCalc",
                               formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('-logM','--logM',help='Log stellar mass of galaxy',type=float,default=None)
    parser.add_argument('-sfr','--sfr',help='Log SFR of galaxy',type=float,default=None)
    parser.add_argument('-logZ','--logZ',help='Log stellar metallicity of galaxy',type=float,default=None)
    parser.add_argument('-z','--z',help='Redshift of galaxy',type=float,default=None)
    parser.add_argument('-i','--i',help='Axis ratio b/a of galaxy',type=float,default=None)
    parser.add_argument('-d2','--d2',help='Diffuse dust optical depth of galaxy',type=float,default=None)
    parser.add_argument('-de','--de',help='Effective dust optical depth of galaxy',type=float,default=None)
    parser.add_argument('-f1','--f1',help='Name of file with one to five columns with independent variable values',type=str,default=None)
    parser.add_argument('-f2','--f2',help='Name of file with diffuse dust optical depths in order to get birth cloud dust optical depths',type=str,default=None)
    parser.add_argument('-pl','--plot',help='Whether or not to plot dust attenuation curve',action='count',default=0)
    parser.add_argument('-bv','--bivar',help='Whether or not to use bivariate model',action='count',default=0)
    parser.add_argument('-eff','--effective',help='Whether to use effective dust attenuation (default is diffuse)',action='count',default=0)
    parser.add_argument('-sa','--samples',help='Number of samples per galaxy desired for dependent variables',type=int,default=50)
    parser.add_argument('-nm','--nummarg',help='Number of Prospector likelihood samples desired for marginalization purposes',type=int,default=20)
    parser.add_argument('-mnp','--max_num_plot',help='Maximum number of plots that will be made in a run',type=int,default=10)
    parser.add_argument('-dir','--img_dir',help='Image directory for created plots (does not have to be created beforehand) ',type=str,default='DustAttnCurves')
    parser.add_argument('-inb','--img_name_base',help='Image name base (minus extension); each plot made will be numbered on top of the image name base',type=str,default=None)

    return parser.parse_args(args=argv)

def regular_grid_interp_scipy(points, values, coords, *, fill_value=None):
    """Perform a linear interpolation in N-dimensions w a regular grid
    The data must be defined on a filled regular grid, but the spacing may be
    uneven in any of the dimensions.
    This implementation uses the ``scipy.interpolate.RegularGridInterpolator`` class which, in turn, is
    based on the implementation from Johannes Buchner's ``regulargrid``
    package https://github.com/JohannesBuchner/regulargrid.
    Args:
        points: A list of vectors with shapes ``(m1,), ... (mn,)``. These
            define the grid points in each dimension.
        values: A tensor defining the values at each point in the grid
            defined by ``points``. This must have the shape
            ``(m1, ... mn, ..., nout)``.
        coords: A matrix defining the coordinates where the interpolation
            should be evaluated. This must have the shape ``(ntest, ndim)``.
    """
    rgi = RGIScipy(points,values,bounds_error=False,fill_value=fill_value)
    return rgi(coords)

def mass_completeness(zred):
    """Uses mass-completeness estimates from Tal+14, for FAST masses
    then applied M_PROSP / M_FAST to estimate Prospector completeness
    Credit: Joel Leja

    Parameters
    ----------
    zred: 1-D Array of redshifts

    Returns
    -------
    Minimum masses in ``log(M*/M_sun)`` for completeness at zred
    """

    zref = np.array([0.65,1,1.5,2.1,3.0])
    mcomp_prosp = np.array([8.71614882,9.07108637,9.63281923,9.79486727,10.15444536])
    return np.interp(zred, zref, mcomp_prosp)

def get_dust_attn_curve_d2(wave,n=0.0,d2=1.0):
    """ Calculate diffuse dust attenuation curve
    Parameters
    ----------
    wave: Float or 1-D Array
        Wavelengths (Angstroms) at which attenuation curve should be evaluated
    n: Float
        Slope parameter of Noll+09 dust attenuation parametrization--difference between true slope and that of Calzetti+00 curve, with positive values signifying shallower curves
    d2: Float
        Diffuse dust optical depth; also referred to as tau throughout this document
    
    Returns
    -------
    Diffuse dust attenuation curve at given wavelength(s) 
    """
    Eb = 0.85 - 1.9*n
    return noll(wave,tau_v=d2,delta=n,c_r=0.0,Ebump=Eb)

def get_dust_attn_curve_d1(wave,d1=1.0):
    """ Calculate birth cloud dust attenuation curve
    Parameters
    ----------
    wave: Float or 1-D Array
        Wavelengths (Angstroms) at which attenuation curve should be evaluated
    d1: Float
        Birth cloud dust optical depth
    
    Returns
    -------
    Birth dust attenuation curve at given wavelength(s); inverse law from Charlot+Fall 00 assumed
    """
    return d1*(wave/5500.0)**(-1)

def getMargSample(num=10,bv=1,eff=1):
    """ Get samples from the Prospector likelihood distribution for proper marginalization

    Parameters
    ----------
    num: Integer
        Number of marginal samples desired
    bv: Boolean
        Whether or not bivariate case for code is being used
    eff: Boolean
        Whether or not single-component dust attenuation model is being used

    Returns
    -------
    Marg_samples: (5 x num) 2D array
        Samples from the likelihood distribution to be used for marginalization
    """
    fn = 'b%de%d.dat'%(bv,eff)
    try:
        stream = pkg_resources.resource_filename('duste',op.join('Marg',fn))
    except:
        stream = op.join('Marg',fn)
    info = np.loadtxt(stream)
    totnum = len(info[0])
    inds = np.random.choice(totnum,num,replace=False)
    return info[:,inds]

def getTraceInfo(trace, bivar=False):
    """ Parse the hierarchical Bayesian model trace object in order to get all samples of the model parameters

    Parameters
    ----------
    trace: Trace object
    bivar: Boolean
        Whether or not the model is bivariate (two dependent variables) or not

    Returns
    -------
    ngrid, taugrid: Multi-D Arrays
        Posterior samples of values of the dependent variables (n and/or tau) at the grid points in the interpolation model (taugrid = None if univariate model)
    log_width, log_width2: 1-D Arrays
        Posterior samples of the log_width parameters (log_width2 = None if univariate), which is a measure of the natural spread in reality around the model valuefs
    rho: 1-D Array
        Posterior samples of the correlation between the errors in ngrid and taugrid (None if univariate)
    """
    ngrid_0, log_width_0 = np.array(getattr(trace.posterior,'ngrid')), np.array(getattr(trace.posterior,'log_width'))
    sh = ngrid_0.shape
    ngrid, log_width = ngrid_0.reshape(sh[0]*sh[1],*sh[2:]), log_width_0.reshape(sh[0]*sh[1])
    if bivar:
        taugrid_0, log_width2_0 = np.array(getattr(trace.posterior,'taugrid')), np.array(getattr(trace.posterior,'log_width2'))
        taugrid, log_width2 = taugrid_0.reshape(sh[0]*sh[1],*sh[2:]), log_width2_0.reshape(sh[0]*sh[1])
        rho_0 = np.array(getattr(trace.posterior,'rho'))
        rho = rho_0.reshape(sh[0]*sh[1])
    else:
        taugrid, log_width2, rho = None, None, None
    return ngrid, log_width, taugrid, log_width2, rho

def getModelSamplesI(xtup, indep_samp, ngrid, log_width, taugrid, log_width2, rho, numsamp=50, nlim=None, taulim=None, return_other=False):
    """ Calculate samples of n and/or tau (the dependent variables of the model) at a given set of points

    Parameters
    ----------
    xtup: List of arrays
        Values of grid points for independent variables in the model; use DustAttnCalc.getPostModelData() for easy generation
    indep_samp: 2-D or 3-D array
        Points at which to evaluate model; variables are differentiated at the outermost dimension; for example, in a 2-D case, each row is a different variable
    ngrid, log_width, taugrid, log_width2, rho: Outputs of getTraceInfo
    numsamp: Integer
        Number of samples desired per galaxy
    nlim, taulim: Two-element 1-D arrays
        Limits for n and tau to keep sample values within reasonable bounds; see DustAttnCalc.make_prop_dict() for values to use; those bounds can be guaranteed only if the independent variables are within the correct bounds as well
    return_other: Boolean
        Whether or not to also return Gaussian width and bivariate correlation parameter values 
    
    Return
    ------
    n_sim, tau_sim: 2-D or 3-D arrays (same dimension as indep_samp but different outer dimension size)
        Samples of n and/or tau at the given points; tau_sim is None if univariate
    width, width2, rho: 1-D Arrays
        Samples of Gaussian width and bivariate correlation if desired
    """
    npts = len(log_width)
    if indep_samp.ndim == 2: indepI = indep_samp.T
    else: indepI = indep_samp.reshape(len(indep_samp),np.prod(indep_samp.shape[1:])).T
    inds = np.random.choice(npts,size=numsamp,replace=False)
    sh = list(indep_samp[0].shape)
    sh.insert(0,numsamp)
    n_sim = np.empty(tuple(sh))
    if taugrid is not None: tau_sim = np.empty(tuple(sh))
    else: tau_sim = None
    for i, ind in enumerate(inds):
        ngrid_mod, width_mod = ngrid[ind], np.exp(log_width[ind])
        r1, r2 = np.random.randn(*n_sim[i].shape), np.random.randn(*n_sim[i].shape)
        interp_part = regular_grid_interp_scipy(xtup, ngrid_mod, indepI).reshape(n_sim[0].shape) # Direct model values (without adding a random component to represent the natural error in the model)
        n_sim[i] = interp_part + width_mod * r1 # Simulated values include the direct interpolated model values plus a random component based on the width parameter
        # Try to ensure that all samples of n are within the desired bounds, but ignore if there are direct model values (without adding a random component) that are considerably below or above the limits
        if np.amin(interp_part)>=nlim[0]-0.1*width_mod and np.amax(interp_part)<=nlim[1]+0.1*width_mod:
            ind_bad = np.where(np.logical_or(n_sim[i]<nlim[0],n_sim[i]>nlim[1]))
            while len(ind_bad[0])>0:
                r1[ind_bad] = np.random.randn(len(ind_bad[0]))
                n_sim[i] = interp_part + width_mod * r1
                ind_bad = np.where(np.logical_or(n_sim[i]<nlim[0],n_sim[i]>nlim[1]))
        # Repeat the same process for tau if bivariate model
        if taugrid is not None: 
            taugrid_mod, width2_mod, rho_mod = taugrid[ind], np.exp(log_width2[ind]), rho[ind]
            interp_part = regular_grid_interp_scipy(xtup, taugrid_mod, indepI).reshape(tau_sim[0].shape)
            tau_sim[i] = interp_part + width2_mod * (rho_mod*r1 + np.sqrt(1.0-rho_mod**2)*r2)
            if np.amin(interp_part)>=taulim[0]-0.1*width2_mod and np.amax(interp_part)<=taulim[1]+0.1*width2_mod:
                ind_bad = np.where(np.logical_or(tau_sim[i]<taulim[0],tau_sim[i]>taulim[1]))
                while len(ind_bad[0])>0:
                    r2[ind_bad] = np.random.randn(len(ind_bad[0]))
                    tau_sim[i] = interp_part + width2_mod * (rho_mod*r1 + np.sqrt(1.0-rho_mod**2)*r2)
                    ind_bad = np.where(np.logical_or(tau_sim[i]<taulim[0],tau_sim[i]>taulim[1]))
    if not return_other: return n_sim, tau_sim
    if taugrid is not None: return n_sim, tau_sim, np.exp(log_width[inds]), np.exp(log_width2[inds]), rho[inds]
    else: return n_sim, tau_sim, np.exp(log_width[inds]), None, None

def plotDustAttn(nvals,tauvals,img_name,wvs,effective=False,label=None):
    """ Plot either diffuse or effective dust attenuation curves

    Parameters
    ----------
    nvals, tauvals: 1-D Arrays
        Samples of n and tau for 1 galaxy only
    img_name: String
        Name for image (can choose desired extension, but if vector graphics format, will need to remove the dpi argument)
    wvs: 1-D Array
        Array of wavelengths (in Angstroms) at which dust attenuation curve will be plotted
    effective: Boolean
        Whether or not effective dust (or diffuse dust if False) is the quantity in question
    label: String
        Text to help describe plot; optimally a list of independent variable values
    """
    fig, ax = plt.subplots()
    lnv, lwv = len(nvals), len(wvs)
    attn_curve = np.empty((lnv,lwv))
    for i in range(lnv):
        attn_curve[i] = get_dust_attn_curve_d2(wvs,n=nvals[i],d2=tauvals[i])
    attn_mean, attn_std = np.mean(attn_curve,axis=0), np.std(attn_curve,axis=0)
    ax.plot(wvs,attn_mean,color='r',linestyle='-',linewidth=2)
    ax.fill_between(wvs,attn_mean-attn_std,attn_mean+attn_std,color='b',alpha=0.1)
    ax.set_xlabel(r'$\lambda$ ($\AA$)')
    if effective: taustr = r"$\hat{\tau}_{\lambda,{\rm eff}}$"
    else: taustr = r"$\hat{\tau}_{\lambda,2}$"
    ax.set_ylabel(taustr)
    ax.set_xlim(min(wvs),max(wvs))
    if label is not None: ax.text(0.4,0.9,label,transform=ax.transAxes,fontsize='x-small')
    fig.savefig(img_name,bbox_inches='tight',dpi=150)

def plotDust12(tau1,tau2,img_name,n,wvs,label=None):
    """ Plot either diffuse or effective dust attenuation curves

    Parameters
    ----------
    tau1, tau2, n: 1-D Arrays
        Samples of birth cloud dust optical depth (tau1), diffuse dust optical depth (tau2), and dust slope index (n) for 1 galaxy only
    img_name: String
        Name for image (can choose desired extension, but if vector graphics format, will need to remove the dpi argument)
    wvs: 1-D Array
        Array of wavelengths (in Angstroms) at which dust attenuation curve will be plotted
    label: String
        Text to help describe plot; optimally a list of independent variable values
    """
    fig, ax = plt.subplots()
    ltau, lwv = len(tau1), len(wvs)
    attn_curve1, attn_curve2 = np.empty((ltau,lwv)), np.empty((ltau,lwv))
    for i in range(ltau):
        attn_curve1[i] = get_dust_attn_curve_d1(wvs,d1=tau1[i])
        attn_curve2[i] = get_dust_attn_curve_d2(wvs,n=n[i],d2=tau2[i])
    attn1_mean, attn1_std = np.mean(attn_curve1,axis=0), np.std(attn_curve1,axis=0)
    attn2_mean, attn2_std = np.mean(attn_curve2,axis=0), np.std(attn_curve2,axis=0)
    ax.plot(wvs,attn1_mean,color='b',linestyle='-',linewidth=2,label=r"$\hat{\tau}_{\lambda,1}$")
    ax.fill_between(wvs,attn1_mean-attn1_std,attn1_mean+attn1_std,color='b',alpha=0.1,label='')
    ax.plot(wvs,attn2_mean,color='r',linestyle='--',linewidth=2,label=r"$\hat{\tau}_{\lambda,2}$")
    ax.fill_between(wvs,attn2_mean-attn2_std,attn2_mean+attn2_std,color='r',alpha=0.1,label='')
    ax.set_xlabel(r'$\lambda$ ($\AA$)')
    ax.set_ylabel(r"$\hat{\tau}_{\lambda}$")
    if label is not None: ax.text(0.15,0.9,label,transform=ax.transAxes,fontsize='x-small')
    ax.set_xlim(min(wvs),max(wvs))
    ax.legend(loc='best',frameon=False)
    fig.savefig(img_name,bbox_inches='tight',dpi=150)

class DustAttnCalc:
    """ Primary mechanism for getting dust attenuation curves """

    def __init__(self,f1=None,f2=None,bv=1,eff=0,samples=50,wv_arr=np.linspace(1500.0,5000.0,501),img_dir_orig='TraceFiles',logM=None,sfr=None,logZ=None,z=None,i=None,d2=None,de=None,nummarg=20):
        """ Initialize the DustAttnCalc Class; independent variables can be passed through files or direct arrays

        Parameters
        ----------
        f1: String (Optional)
            Name of file with independent variable values
        f2: String (Optional)
            Name of file with diffuse dust values in order to get birth cloud dust values
        bv: Boolean
            Whether or not a bivariate model is desired
        eff: Boolean
            Whether or not the model should be for effective or diffuse dust
        samples: Integer
            Number of samples per galaxy desired for dependent variables
        wv_arr: 1-D Array
            Wavelengths (Angstroms) at which dust attenuation curves should be calculated
        img_dir_orig: String
            Name of directory with the trace netcdf and data files: default in the package is TraceFiles
        logM, sfr, logZ, z, i, d2, de: 1-D (or possibly 2-D) arrays, Floats, or NoneType (Optional)
            Values of independent variables that you have; any subset is allowed, but d2 and de are reserved for univariate models (for diffuse and effective dust, respectively)
        nummarg: Integer
            Number of Prospector likelihood samples desired for marginalization purposes
        """
        self.input_file, self.d2_file = f1, f2
        self.bivar, self.effective = bv, eff
        if self.bivar: 
            self.props = np.array(['logM','sfr','logZ','z','i'])
        else:
            if self.effective: self.props = np.array(['logM','sfr','logZ','z','de'])
            else: self.props = np.array(['logM','sfr','logZ','z','d2'])
        full_arr = [wv_arr, logM, sfr, logZ, z, i, d2, de]
        full_names = ['wv', 'logM', 'sfr', 'logZ', 'z', 'i', 'd2', 'de']
        for arr, name in zip(full_arr, full_names):
            if type(arr)==float or type(arr)==int: val = np.array([arr])
            elif type(arr)==list: val = np.array(arr)
            else: val = arr
            setattr(self,name+'_arr',val)
        if f1 is not None: self.read_indep_file()
        if f2 is not None: self.read_d2_file()
        self.samples = samples
        self.img_dir_orig = img_dir_orig
        self.nummarg = nummarg
        self.make_prop_dict()
        self.label_creation()

    def label_creation(self):
        """ Create appropriate labels for plots and bookkeeping; called during initialization of DustAttnCalc Class """
        for prop in self.props: setattr(self,prop,1)
        self.n = 1
        indepext = np.array(['m','sfr','logZ','z','i','d1','d2','de'])
        self.indep_pickle_name, self.indep_name, self.indep_lab = [], [], []
        self.dataname=''
        for i, name in enumerate(self.prop_dict['names'].keys()):
            try:
                if getattr(self,name):
                    self.indep_pickle_name.append(self.prop_dict['pickle_names'][name])
                    self.indep_name.append(self.prop_dict['names'][name])
                    self.indep_lab.append(self.prop_dict['labels'][name])
                    self.dataname+=indepext[i]
            except:
                pass
        self.indep_pickle_name, self.indep_name = np.array(self.indep_pickle_name), np.array(self.indep_name)
        if self.effective: nstr, taustr = 'ne', 'de'
        else: nstr, taustr = 'n', 'tau2'
        if self.bivar: self.dataname+=f'n_d2'
        else:
            if self.n: self.dataname+=f'_n'
            else: self.dataname+=f'_d2'
        self.extratext = self.dataname
        if self.n: 
            self.dep_name, self.dep_lab = self.dep_dict['names'][nstr], self.dep_dict['labels'][nstr]
        else:
            self.dep_name, self.dep_lab = self.dep_dict['names'][taustr], self.dep_dict['labels'][taustr]
        self.nlim = np.array([self.dep_dict['min'][nstr],self.dep_dict['max'][nstr]])
        self.taulim = np.array([self.dep_dict['min'][taustr],self.dep_dict['max'][taustr]])
        self.nstr, self.taustr = nstr, taustr
        
    def make_prop_dict(self):
        """ Create dictionaries with names, labels, and reasonable limits for parameters; called during initialization of DustAttnCalc Class """
        self.prop_dict, self.dep_dict = {}, {}
        self.prop_dict['names'] = {'logM': 'logM', 'sfr': 'logSFR', 'logZ': 'logZ', 'z': 'z', 'i':'axis_ratio', 'd1':'dust1', 'd2':'dust2', 'de':'duste'}
        self.dep_dict['names'] = {'tau2': 'dust2', 'n': 'n', 'de': 'duste', 'ne': 'ne'}
        self.prop_dict['labels'] = {'logM': r'$\log M_*$', 'sfr': r'$\log$ SFR$_{\rm{100}}$', 'logZ': r'$\log (Z/Z_\odot)$', 'z': 'z', 'i':r'$b/a$', 'd1':r"$\hat{\tau}_{1}$", 'd2':r"$\hat{\tau}_{2}$", 'de':r"$\hat{\tau}_{\rm {eff}}$"}
        self.dep_dict['labels'] = {'tau2': r"$\hat{\tau}_{2}$", 'n': 'n', 'de': r"$\hat{\tau}_{\rm {eff}}$", 'ne': r"$n_{\rm {eff}}$"}
        self.prop_dict['pickle_names'] = {'logM': 'stellar_mass', 'sfr': 'sfr_100', 'logZ': 'log_z_zsun', 'z': 'z', 'i':'inc', 'd1':'dust1', 'd2':'dust2', 'de':'tau_eff'}
        self.dep_dict['pickle_names'] = {'tau2': 'dust2', 'n': 'dust_index', 'de': 'tau_eff', 'ne': 'n_eff'}
        self.prop_dict['min'] = {'logM': 8.74, 'sfr': -2.06, 'logZ': -1.70, 'z': 0.51, 'i': 0.09, 'd1': 0.01, 'd2': 0.01, 'de': 0.01}
        self.prop_dict['max'] = {'logM': 11.30, 'sfr': 2.11, 'logZ': 0.18, 'z': 2.83, 'i': 0.97, 'd1': 2.23, 'd2': 1.95, 'de': 2.19}
        self.dep_dict['min'] = {'tau2': 0, 'n': -1.0, 'de': 0, 'ne': -1.4}
        self.dep_dict['max'] = {'tau2': 4.0, 'n': 0.4, 'de': 5.0, 'ne': 0.6}

    def read_indep_file(self):
        """ Read independent variable file for univariate or bivariate models; called during initialization of DustAttnCalc Class """
        dat = Table.read(self.input_file, format='ascii')
        for prop in self.props:
            if prop in dat.colnames: setattr(self,prop+'_arr',dat[prop])
            else: setattr(self,prop+'_arr',None)
    
    def read_d2_file(self):
        """ Read independent variable file for diffuse dust values which can be used in generating birth cloud dust values; called during initialization of DustAttnCalc Class """
        if self.input_file is None: 
            dat = Table.read(self.d2_file, format='ascii')
            self.d2_arr = dat['d2']
        else: print("Independent file was already defined!")

    def get_indep(self,indep=None,prop_list=None):
        """ Figure out which indices are included as independent variables and which ones need to be marginalized over: called by run_dust_modules, the primary method in the class.
        """
        mins, maxs = self.get_indep_lims()
        inds, inds_not = [], []
        mins_not, maxs_not = [], []
        if indep is not None: assert prop_list is not None, "Didn't provide variable name list!"
        for i, prop in enumerate(self.props):
            prop_arr = getattr(self,prop+'_arr')
            if indep is None: cond = prop_arr is not None
            else: cond = prop in prop_list
            if cond: 
                inds.append(i)
                if indep is None: self.ngal = len(prop_arr)
                else: self.ngal = len(indep[i])
            else: 
                print("Marginalizing over %s"%(prop))
                inds_not.append(i)
                mins_not.append(mins[prop])
                maxs_not.append(maxs[prop])
        inds, inds_not = np.array(inds), np.array(inds_not)
        return inds, inds_not, mins_not, maxs_not

    def run_dust_modules(self,indep=None,prop_list=None):
        """ Run all necessary modules to provide values of the dependent variables given the independent variables. The independent variable list can be provided directly or taken from initialization.
        
        Parameters (Optional)
        ---------------------
        indep: 2-D (or possibly 3-D) array
            Array of points in the independent parameter space, with variable distinction on the outer dimension; can be just a subset of the variables used in the model itself. For example, to include just logM and SFR, you can make a two-row array with values of logM and SFR and indicate the proper labels with prop_list
        prop_list: List of strings
            Names of variables corresponding to indep; options are 'logM' (stellar mass), 'sfr' (current star formation rate), 'logZ' (stellar metallicity), 'z' (redshift), 'i' (inclination), 'd2' (diffuse dust optical depth), 'de' (effective dust optical depth)
        """
        trace, xtup = self.getPostModelData()
        ngrid, log_width, taugrid, log_width2, rho = getTraceInfo(trace, bivar=self.bivar)
        if not self.bivar and not self.n: nlim = self.taulim
        else: nlim = self.nlim
        marg_samples = getMargSample(num=self.nummarg, bv=self.bivar, eff=self.effective)
        inds, inds_not, mins_not, maxs_not = self.get_indep(indep,prop_list)
        self.inds = inds
        if indep is None or not np.array_equiv(prop_list,self.props): self.indep_samp = np.empty((len(self.props),self.ngal))   
        else: self.indep_samp = np.empty_like(indep) 
        for i, prop in enumerate(self.props[inds]):
            if indep is None: self.indep_samp[inds[i]] = getattr(self,prop+'_arr')
            else: self.indep_samp[inds[i]] = indep[i]
        n_sim, ws = 0., 0.
        if self.bivar: tau_sim, w2s, rs = 0., 0., 0.
        else: tau_sim, w2s, rs = None, None, None
        for i in range(self.nummarg):
            for ind_not in inds_not: self.indep_samp[ind_not] = marg_samples[ind_not,i]
            n_simi, tau_simi, wsi, w2si, rsi = getModelSamplesI(xtup, self.indep_samp, ngrid, log_width, taugrid, log_width2, rho, nlim=nlim, taulim=self.taulim, return_other=True, numsamp=self.samples)
            n_sim += n_simi; ws += wsi
            if self.bivar: tau_sim += tau_simi; w2s += w2si; rs += rsi
        n_sim /= self.nummarg; ws /= self.nummarg
        if self.bivar: tau_sim /= self.nummarg; w2s /= self.nummarg; rs /= self.nummarg
        return n_sim, tau_sim, ws, w2s, rs

    def get_d1(self,d2,totnum=1001,tau_sim=None):
        """ Get birth cloud dust optical depth from diffuse dust optical depth

        Parameters
        ----------
        d2: 1-D Array
            Diffuse dust optical depths at which you want the birth cloud dust optical depths
        totnum: Integer
            Internal parameter; no need for adjustment
        tau_sim: 2-D (or possibly 3-D) array
            Optional array of posterior diffuse dust optical depth values (for the dust1-dust2 hierarchical Bayesian model) to save time in the calculation; used when several iterations of get_d1 are required in plotDust()
        
        Return
        ------
        d1ret: 2-D array
            Samples of birth cloud dust optical depth corresponding to d2
        tau_sim: 2-D (or possibly 3-D) array
            Posterior samples of diffuse dust from the dust1-dust2 model; returned if tau_sim wasn't already provided to the function
        """
        ext, indep_name = self.extratext, self.indep_name
        self.extratext, self.indep_name = 'd1_d2', ['dust1']
        d1min, d1max = self.prop_dict['min']['d1'], self.prop_dict['max']['d1']
        indep = np.linspace(d1min,d1max,totnum)[None,:]
        if tau_sim is None: 
            already_calc=0
            trace, xtup = self.getPostModelData()
            ngrid, log_width, _, _, _ = getTraceInfo(trace, bivar=False)
            tau_sim, _ = getModelSamplesI(xtup, indep, ngrid, log_width, None, None, None, nlim=self.taulim, numsamp=self.samples)
        else:
            already_calc=1
        d1ret = np.empty((len(tau_sim),len(d2)))
        for i in range(len(tau_sim)):
            indsort = np.argsort(tau_sim[i])
            d1ret[i] = np.interp(d2,tau_sim[i][indsort],indep[0][indsort])
        self.extratext, self.indep_name = ext, indep_name
        if already_calc: return d1ret
        return d1ret, tau_sim

    def getPostModelData(self):
        """ Read appropriate .netcdf and .dat files given model requirements to initiate the trace parsing process
        
        Return
        ------
        trace: Trace object
        xtup: List of arrays
            Unique values of coordinates of grid points in each parameter dimension used in the given hierarchical Bayesian model; the true grid is created using np.meshgrid(*xtup)
        """
        try:
            globsearch = '*%s*eff_%d*.nc'%(self.extratext,self.effective)
            dirinfo = pkg_resources.resource_listdir('duste',self.img_dir_orig)
            nclist_orig = fnmatch.filter(dirinfo,globsearch)[0]
            nclist = pkg_resources.resource_filename('duste',op.join(self.img_dir_orig,nclist_orig))
        except:
            globsearch = op.join(self.img_dir_orig,'*%s*eff_%d*.nc'%(self.extratext,self.effective))
            nclist = glob(globsearch)[0]
        trace = az.from_netcdf(nclist)
        print("Trace file:", nclist)
        try:
            globsearch = '*%s*eff_%d*.dat'%(self.extratext,self.effective)
            dirinfo = pkg_resources.resource_listdir('duste',self.img_dir_orig)
            datf_orig = fnmatch.filter(dirinfo,globsearch)[0]
            datf = pkg_resources.resource_filename('duste',op.join(self.img_dir_orig,datf_orig))
        except:
            globsearch = op.join(self.img_dir_orig,'*%s*eff_%d*.dat'%(self.extratext,self.effective))
            datf = glob(globsearch)[0]
        dat = Table.read(datf,format='ascii')
        print("Dat file:",datf)
        ndim = len(self.indep_name)
        name_0 = self.indep_name[0]
        temp = np.unique(dat[name_0])
        grid_len = temp.size
        print("Measured grid length from file:", grid_len)
        x = np.empty((ndim,grid_len))
        for i, name in enumerate(self.indep_name):
            x[i] = np.unique(dat[name])
        return trace, tuple(x)

    def calcDust(self,indep=None,prop_list=None,img_name=None,img_dir='DustAttnCurves',max_num_plot=10,plot_tau=True):
        """ Plot dust attenuation curves: this can be either a combination of diffuse and birth cloud dust or just diffuse/effective birth cloud dust.

        Parameters
        ----------
        indep, prop_list (optional): See run_dust_modules for description
        img_name: String
            Name for image; a basic name is given in case none is provided
        img_dir: String
            Name for image directory; default is DustAttnCurves
        max_num_plot: Integer
            In case several points in the independent parameter space are provided, this is the maximum number of plots that will be made (to avoid inundation of files)
        plot_tau: Boolean
            Whether dust should be plotted (no plots will be made if this is False or 0)

        Return
        ------
        dac, dac1: Arrays of same dimension of indep (but different shape); 1st dimension is different samples from the model; 2nd to 2nd last dimension (typically just 1 dimension in total) correspond to the different galaxies; last dimension corresponds to the wavelength array
            Dust attenuation curve as a function of wavelength (set at self.wv_arr, which is set at initialization)
        navg, d2avg, d1avg: Arrays of average parameter values (n/n_eff, tau_2/tau_eff, and tau1)--dim 2 fewer than dac
        nerr, d2err, d1err: Arrays of standard deviations of parameter values (n/n_eff, tau_2/tau_eff, and tau1)--dim 2 fewer than dac
        
        Note: dac1, d1avg, and d1err will be NoneType if the effective dust model is being used.
        """
        mkpath(img_dir)
        if img_name is None:
            img_name_base = 'DustAttnCurve'
            img_name = f'{img_name_base}_bv_{self.bivar}_eff_{self.effective}'
        img_name_full = op.join(img_dir,img_name)
        nvals, tauvals, ws, w2s, rs = self.run_dust_modules(indep,prop_list)
        if not self.bivar: tauvals = np.repeat(self.indep_samp[-1][None,:],len(nvals),axis=0)
        sh = list(nvals.shape); sh.insert(len(sh),len(self.wv_arr)); sh = tuple(sh)
        dac = np.empty(sh)
        if not self.effective: dac1 = np.empty(sh)
        else: dac1, d1avg, d1err = None, None, None

        inds = np.random.choice(len(nvals[0]),min(max_num_plot,len(nvals[0])),replace=False)
        if not self.effective:
            d1vals = np.empty_like(nvals)
            _, tau_sim = self.get_d1(tauvals[0])
            for i in range(len(nvals)): d1vals[i] = np.average(self.get_d1(tauvals[i], tau_sim=tau_sim),axis=0)

        for i, index in enumerate(np.ndindex(nvals.shape)):
            dac[index] = get_dust_attn_curve_d2(self.wv_arr,n=nvals[index],d2=tauvals[index])
            if not self.effective: dac1[index] = get_dust_attn_curve_d1(self.wv_arr, d1=d1vals[index])

        for i, ind in enumerate(inds):
            label = ''
            for j, indprop in enumerate(self.inds):
                label += self.indep_lab[indprop] + r'$=%.2f$; '%(self.indep_samp[indprop,ind])
            if plot_tau and self.effective: plotDustAttn(nvals[:,ind],tauvals[:,ind],img_name_full+'%02d'%(i),self.wv_arr,self.effective,label)
            if plot_tau and not self.effective: plotDust12(d1vals[:,ind],tauvals[:,ind],img_name_full+'_%02d'%(i),nvals[:,ind],self.wv_arr,label)
        navg, tauavg = np.average(nvals, axis=0), np.average(tauvals, axis=0)
        nerr, tauerr = np.std(nvals, axis=0), np.std(tauvals, axis=0)
        if not self.effective: d1avg, d1err = np.average(d1vals, axis=0), np.std(d1vals, axis=0)
        if navg.size==1:
            if self.effective: print(r"n_eff, tau_eff: %.3f +/- %.3f, %.3f +/- %.3f"%(navg[0],nerr[0],tauavg[0],tauerr[0]))
            else: print(r"n, tau_2, tau_1: %.3f +/- %.3f, %.3f +/- %.3f, %.3f +/- %.3f$"%(navg[0],nerr[0],tauavg[0],tauerr[0],d1avg[0],d1err[0]))
        else:
            print("n:", navg); print("n error:", nerr)
            print("tau:", tauavg); print("tau error:", tauerr)
            if not self.effective: 
                print("tau1:", d1avg); print("tau1 error:", d1err)
        if self.bivar: print(r"n vs tau Average Correlation Coefficient rho: %0.3f"%(np.average(rs)))
        return dac, dac1, navg, tauavg, d1avg, nerr, tauerr, d1err
                
    def get_indep_lims(self):
        """ Print and return reasonable mins and maxs for independent parameters """
        mins, maxs = self.prop_dict['min'], self.prop_dict['max']
        print("Mins:", mins)
        print("Maxs:", maxs)
        return mins, maxs

    def get_like(self,dep,indep=None,prop_list=None):
        """ Get likelihood of given sample of n and/or tau

        Parameters
        ----------
        dep: 1-D, 2-D, or 3-D array
            Values of dependent variable(s); if bivariate model desired, distinction between n and tau should be done in the outermost dimension of the array: for example, a row of n values and a row of tau values.
        indep, prop_list: See run_dust_modules for description

        Return
        ------
        like: Same size array as dep (unless bivariate case, in which one fewer dimension)
            Array of likelihoods
        """
        nsim, tausim, ws, w2s, rs = self.run_dust_modules(indep,prop_list)
        like = np.empty(nsim.shape)
        if not self.bivar: 
            for i in range(len(like)):
                like[i] = 1.0/(ws[i]*np.sqrt(2.0*np.pi)) * np.exp(-0.5*(dep-nsim[i])**2/ws[i]**2)
        else:
            for i in range(len(like)):
                z = (dep[0]-nsim[i])**2/ws[i]**2 + (dep[1]-tausim[i])**2/w2s[i]**2 - 2.0*rs[i]*(dep[0]-nsim[i])*(dep[1]-tausim[i])/(ws[i]*w2s[i])
                like[i] = 1.0/(2.0*np.pi*ws[i]*w2s[i]*np.sqrt(1.0-rs[i]**2)) * np.exp(-0.5*z/(1.0-rs[i]**2))
        return np.median(like,axis=0)

    def get_weights(self,dep,indep=None,prop_list=None):
        """ Get weights of samples of dependent variables (observations) based on the hierarchical Bayesian dust model vs Prospector priors. See get_like for description of arguments """
        like = self.get_like(dep,indep,prop_list)
        myclip_a, myclip_b, my_mean, my_std = self.dep_dict['min'][self.taustr], self.dep_dict['max'][self.taustr], 0.3, 1.0
        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        n_wid = self.dep_dict['max'][self.nstr]-self.dep_dict['min'][self.nstr]
        if self.bivar: 
            prprior = truncnorm.pdf(dep[1],a,b,loc=my_mean,scale=my_std) / n_wid
            weight = like / prprior
        else:  
            weight = like*n_wid
        return weight

if __name__=='__main__':
    args = parse_args()
    dobj = DustAttnCalc(f1=args.f1, f2=args.f2, bv=args.bivar, eff=args.effective, samples=args.samples, logM=args.logM, sfr=args.sfr, logZ=args.logZ, z=args.z, i=args.i, d2=args.d2, de=args.de, nummarg=args.nummarg)
    if args.f1 is None and args.f2 is not None: 
        d1sim, _ = dobj.get_d1(dobj.d2_arr)
        d1, d1e = np.average(d1sim,axis=0), np.std(d1sim,axis=0)
        print("Dust1 average values:", d1)
        print("Dust1 Standard deviations:", d1e)
    else: 
        dac, dac1, n, tau, tau1, ne, taue, tau1e = dobj.calcDust(img_dir=args.img_dir, max_num_plot=args.max_num_plot, plot_tau=args.plot, img_name=args.img_name_base)