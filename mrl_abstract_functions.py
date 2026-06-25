import os
import sys
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import rc
font = {'size'   : 25}
rc('font', **font)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
rc('text', usetex=True)
plt.rcParams['font.family']='Computer Modern'
plt.rcParams['figure.figsize'] = (8,8)


from scipy.stats import multivariate_normal, norm, bernoulli, poisson, expon, uniform, truncexpon, truncnorm
from sklearn.calibration import calibration_curve

import iminuit
from iminuit import cost, Minuit

class MyPDF():
    def __init__(self,function,ndim):
        self.function = function
        self.ndim = ndim
    def pdf(self,x):
        if len(x.shape) == 1:
            return self.function.pdf(x)
        else:
            return np.prod(np.array([self.function.pdf(xval) for xval in x.T]),0)
    def rvs(self,size):
        x =  np.array([self.function.rvs(size=size) for _ in range(self.ndim)]).T
        if self.ndim == 1:
            x = x[:,0]
        return x

def create_dataset(background_model,signal_model,S=1000,B=1000,dim=1,seed=42):
    np.random.seed(seed)
    if S > 0:
        NS = poisson(S).rvs()
    else:
        NS = 0
    NB = poisson(B).rvs()
    N = NS+NB
    if dim == 1:
        X = np.zeros(N)
    else:
        X = np.zeros((N,dim))
    y = np.zeros(N)
    if NS > 0:
        labels = np.random.choice(np.arange(N),p=np.ones(N)/N,size=NS,replace=False)
        y[labels] = 1
        X_sig = signal_model.rvs(size=NS)
        X[y==1] = X_sig
    X_bkg = background_model.rvs(size=NB)
    X[y==0] = X_bkg

    return X,y

class Analysis():
    def __init__(self,signal_model,background_model, biased_background_model,ndim,results_folder):
        self.signal_model = signal_model
        self.background_model = background_model
        self.biased_background_model = biased_background_model
        self.ndim = ndim
        self.results_folder = results_folder
        os.makedirs(str(self.results_folder), exist_ok=True)


    def density(self, x, mu, Bfit):
        if len(x.shape) == 1:
            return mu + Bfit, (
                mu * self.signal_model.pdf(x) + Bfit * self.background_model.pdf(x)
            )
        else:
            return mu + Bfit, (
                mu * self.signal_model.pdf(x.T) + Bfit * self.background_model.pdf(x.T)
            )

    def density_bias_with_cut(self, x, mu, Bfit, eff_bkg, eff_sig, eff_cr):
        if len(x.shape) == 1:
            return mu  *  eff_sig  + eff_cr * Bfit, (
                mu  * self.signal_model.pdf(x)   +   Bfit * eff_cr * self.biased_background_model.pdf(x) / eff_bkg
            )
        else:
            return mu  *  eff_sig  + eff_cr * Bfit, (
                mu  * self.signal_model.pdf(x.T)   +   Bfit * eff_cr * self.biased_background_model.pdf(x.T) / eff_bkg
            )
        
    def experiment(self,S,B,CRsize,Toysize,seed,eps=0.01,show_plots = False,ndim=1,density_plot=True):

        ### Create three datasets
        X,y = create_dataset(background_model=self.background_model,signal_model=self.signal_model,S=S,B=B,seed=seed,dim=ndim)

        if show_plots and ndim == 1:
            _, bins, _ = plt.hist(X, bins=100, alpha=0.5,histtype='step', label='Full', density=True)
            plt.hist(X[y==0], bins=bins, alpha=0.5,histtype='step', label='Background', density=True)
            plt.plot(0.5*(bins[1:]+bins[:-1]),self.background_model.pdf(0.5*(bins[1:]+bins[:-1])),label='Background model')
            plt.plot(0.5*(bins[1:]+bins[:-1]),self.biased_background_model.pdf(0.5*(bins[1:]+bins[:-1])),label='Biased background model')
            plt.hist(X[y==1], bins=bins, alpha=0.5,histtype='step', label='Signal', density=True)
            plt.plot(0.5*(bins[1:]+bins[:-1]),self.signal_model.pdf(0.5*(bins[1:]+bins[:-1])),label='Signal model')
            plt.xlabel(r'$x$')
            plt.ylabel('Density')
            plt.title('Signal and Background Distributions')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/data_distributions.pdf')
            plt.show()

        if show_plots and ndim == 2:
            _, binsx, binsy, _ = plt.hist2d(X[:,0],X[:,1],bins=100, density=True)
            plt.ylabel(r'$x_2$')
            plt.xlabel(r'$x_1$')
            plt.title('Data Distribution')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/data_distribution.pdf')
            plt.show()

            plt.hist2d(X[y==1,0],X[y==1,1],bins=[binsx,binsy], density=True)
            plt.ylabel(r'$x_2$')
            plt.xlabel(r'$x_1$')
            plt.title('Signal Distribution')
            plt.colorbar()
            plt.show()

            plt.hist2d(X[y==0,0],X[y==0,1],bins=[binsx,binsy], density=True)
            plt.ylabel(r'$x_2$')
            plt.xlabel(r'$x_1$')
            plt.title('Background Distribution')
            plt.colorbar()
            plt.show()
        XCR,yCR = create_dataset(background_model=self.background_model,signal_model=self.signal_model,S=0,B=CRsize,seed=seed,dim=ndim)
        XToy,yToy = create_dataset(background_model=self.biased_background_model,signal_model=self.signal_model,S=Toysize,B=Toysize,seed=seed,dim=ndim)

        # print(X.shape,y.shape,XCR.shape,yCR.shape,XToy.shape,yToy.shape)
        if ndim == 1:
            c = cost.ExtendedUnbinnedNLL(X, self.density)
        else:
            c = cost.ExtendedUnbinnedNLL(X.T, self.density)

        m = Minuit(c, mu=1, Bfit=B)
        #m.limits["mu"] = (0, None)
        m.migrad()
        m.minos()

        central_value_unbiased = m.values['mu']
        error_unbiased = m.errors['mu']
        merror_unbiased = [float(repr(m.merrors['mu']).split('lower')[1].split('=')[1].split('upper')[0]),float(repr(m.merrors['mu']).split('upper')[1].split('=')[1].split('is_valid')[0])]
        Bfit_val = m.values['Bfit']
        Bfit_error = m.errors['Bfit']
        if ndim == 1:
            norm1 = self.density(X,S,B)[0]
            norm2 = self.density(X,*m.values)[0]
            llr_unbiased = -2*np.sum(np.log(self.density(X,S,B)[1]/norm1)-np.log(self.density(X,*m.values)[1]/norm2))
        else:
            norm1 = self.density(X.T,S,B)[0]
            norm2 = self.density(X.T,*m.values)[0]
            llr_unbiased = -2*np.sum(np.log(self.density(X.T,S,B)[1]/norm1)-np.log(self.density(X.T,*m.values)[1]/norm2))

        if show_plots:
            grid_unbiased, profile_unbiased, bool_unbiased = m.mnprofile('mu')

        if show_plots and ndim == 1:

            yields, bins, edges = plt.hist(X,density=density_plot,histtype='step',bins=100,label='Data');
            vals = 0.5*(bins[1:]+bins[:-1])
            bin_size = bins[1]-bins[0]
            if density_plot:
                plt.plot(vals,self.density(vals,m.values['mu'],m.values['Bfit'])[1]/self.density(vals,m.values['mu'],m.values['Bfit'])[0],label=f'$\mu_s$={central_value_unbiased/S:.3f} +/- {error_unbiased/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}')
            else:
                plt.plot(vals,bin_size*self.density(vals,m.values['mu'],m.values['Bfit'])[1],label=f'$\mu_s$={central_value_unbiased/S:.3f} +/- {error_unbiased/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}')
            #
            plt.legend(loc='best')
            plt.ylabel('Events')
            plt.xlabel(r'$x$')
            plt.title(f'Unbiased fit, $\lambda_b ={round(B)}$, $\lambda_s={round(S)}$')
            plt.xlim(-4,4)
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/fit_unbiased.pdf')

            plt.show()

        if show_plots and ndim == 2:

            yields, binsx, binsy, edges = plt.hist2d(X[:,0],X[:,1],density=density_plot,bins=[100,100],label='Data');
            valsx = 0.5*(binsx[1:]+binsx[:-1])
            valsy = 0.5*(binsy[1:]+binsy[:-1])

            bin_size_x = binsx[1]-binsx[0]
            bin_size_y = binsy[1]-binsy[0]

            # valsx_contour, valsy_contour = np.meshgrid(valsx,valsy)
            Z = np.zeros((len(valsy),len(valsx)))
            for i in range(len(valsy)):
                for j in range(len(valsx)):
                    val = np.array([valsx[j],valsy[i]]).reshape(-1,1)
                    # print(i,j,val,density(val,m.values['mu'],m.values['Bfit']))
                    if density_plot:
                        Z[i,j] = (self.density(val,m.values['mu'],m.values['Bfit'])[1]/self.density(val,m.values['mu'],m.values['Bfit'])[0])
                    else:
                        Z[i,j] = (bin_size_x*bin_size_y* self.density(val,m.values['mu'],m.values['Bfit'])[1])

            plt.contour(valsx,valsy,Z,levels=10,cmap='plasma')
            # plt.colorbar()
            # plt.legend(loc='best')
            plt.text(0.05, 0.95, f'$\mu_s$={central_value_unbiased/S:.3f} +/- {error_unbiased/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}', transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.ylabel(r'$x_2$')
            plt.xlabel(r'$x_1$')
            plt.title(f'Unbiased fit, $\lambda_b ={round(B)}$, $\lambda_s={round(S)}$')
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/fit_unbiased.pdf')

            plt.show()



        if ndim == 1:
            c = cost.ExtendedUnbinnedNLL(X, self.density_bias_with_cut)
        else:
            c = cost.ExtendedUnbinnedNLL(X.T, self.density_bias_with_cut)

        eff_bkg = 1.0
        eff_sig = 1.0
        eff_cr = 1.0

        m = Minuit(c, mu=1, Bfit=B,eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)
        #m.limits["mu"] = (0, None)
        ### I could add penalty terms here
        m.fixed["eff_bkg"] = True
        m.fixed["eff_sig"] = True
        m.fixed["eff_cr"] = True
        m.migrad()
        m.minos()


        central_value_biased = m.values['mu']
        error_biased = m.errors['mu']
        merror_biased = [float(repr(m.merrors['mu']).split('lower')[1].split('=')[1].split('upper')[0]),float(repr(m.merrors['mu']).split('upper')[1].split('=')[1].split('is_valid')[0])]  
        Bfit_val = m.values['Bfit']
        Bfit_error = m.errors['Bfit']

        if ndim == 1:
            norm1 = self.density_bias_with_cut(X,S,B,eff_bkg, eff_sig, eff_cr)[0]
            norm2 = self.density_bias_with_cut(X,*m.values)[0]
            llr_biased = -2*np.sum(np.log(self.density_bias_with_cut(X,S,B,eff_bkg, eff_sig, eff_cr)[1]/norm1)-np.log(self.density_bias_with_cut(X,*m.values)[1]/norm2))
        else:
            norm1 = self.density_bias_with_cut(X.T,S,B,eff_bkg, eff_sig, eff_cr)[0]
            norm2 = self.density_bias_with_cut(X.T,*m.values)[0]
            llr_biased = -2*np.sum(np.log(self.density_bias_with_cut(X.T,S,B,eff_bkg, eff_sig, eff_cr)[1]/norm1)-np.log(self.density_bias_with_cut(X.T,*m.values)[1]/norm2))

        if show_plots:
            grid_biased, profile_biased, bool_biased = m.mnprofile('mu')   

        if show_plots and ndim == 1:
            yields, bins, edges = plt.hist(X,density=density_plot,histtype='step',bins=100,label='Data');
            vals = 0.5*(bins[1:]+bins[:-1])
            bin_size = bins[1]-bins[0]
            if density_plot:
                plt.plot(vals,self.density_bias_with_cut(vals,mu=m.values['mu'], Bfit=m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[1]/self.density_bias_with_cut(vals,mu=m.values['mu'], Bfit=m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[0],label=f'$\mu_s$={central_value_biased/S:.3f} +/- {error_biased/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}')
            else:
                plt.plot(vals,bin_size*self.density_bias_with_cut(vals,mu=m.values['mu'], Bfit=m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[1],label=f'$\mu_s$={central_value_biased/S:.3f} +/- {error_biased/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}')
            # 
            plt.legend(loc='best')
            plt.ylabel('Events')
            plt.xlabel(r'$x$')  
            plt.title(f'Biased fit, $\lambda_b ={round(B)}$, $\lambda_s={round(S)}$')              
            plt.xlim(-4,4)
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/fit_biased.pdf')
            plt.show()

        if show_plots and ndim == 2:

            yields, binsx, binsy, edges = plt.hist2d(X[:,0],X[:,1],density=density_plot,bins=[100,100],label='Data');
            valsx = 0.5*(binsx[1:]+binsx[:-1])
            valsy = 0.5*(binsy[1:]+binsy[:-1])

            bin_size_x = binsx[1]-binsx[0]
            bin_size_y = binsy[1]-binsy[0]

            # valsx_contour, valsy_contour = np.meshgrid(valsx,valsy)
            Z = np.zeros((len(valsy),len(valsx)))
            for i in range(len(valsy)):
                for j in range(len(valsx)):
                    val = np.array([valsx[j],valsy[i]]).reshape(-1,1)
                    # print(i,j,val,density(val,m.values['mu'],m.values['Bfit']))
                    if density_plot:
                        Z[i,j] = (self.density_bias_with_cut(val,m.values['mu'],m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[1]/self.density_bias_with_cut(val,m.values['mu'],m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[0])
                    else:
                        Z[i,j] = (bin_size_x*bin_size_y* self.density_bias_with_cut(val,m.values['mu'],m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[1])

            plt.contour(valsx,valsy,Z,levels=10,cmap='plasma')
            # plt.colorbar()
            # plt.legend(loc='best')
            plt.text(0.05, 0.95, f'$\mu_s$={central_value_biased/S:.3f} +/- {error_biased/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}', transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.ylabel(r'$x_2$')
            plt.xlabel(r'$x_1$')

            plt.title(f'Biased fit, $\lambda_b ={round(B)}$, $\lambda_s={round(S)}$')
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/fit_biased.pdf')

            plt.show()        

        likelihoods_cr = self.biased_background_model.pdf(XCR)
        likelihoods = self.biased_background_model.pdf(X)
        likelihoods_toy = self.biased_background_model.pdf(XToy[yToy==0])
        likelihoods_signal_toy = self.biased_background_model.pdf(XToy[yToy==1])

        npoints = 100
        cuts = [np.quantile(likelihoods_toy,i/npoints) for i in range(npoints+1)]    

        fractions_toy, bins_toy = np.histogram(likelihoods_toy,bins=cuts,density=False)
        # print(fractions_toy.shape,likelihoods_toy.shape)
        # print(np.sum(fractions_toy),len(likelihoods_toy))
        fractions_toy=fractions_toy/np.sum(fractions_toy)
        fractions, bins = np.histogram(likelihoods_cr,bins=cuts,density=False)
        fractions=fractions/np.sum(fractions)

        vals = 0.5*(bins[1:]+bins[:-1])


        accumulated_fractions_toy = np.array([np.sum(fractions_toy[:i+1]) for i in range(len(fractions_toy))])
        accumulated_fractions = np.array([np.sum(fractions[:i+1]) for i in range(len(fractions_toy))])

        if np.max(np.abs(accumulated_fractions_toy-accumulated_fractions)) > eps:
            Lmin = vals[np.argmax(np.abs(accumulated_fractions_toy-accumulated_fractions))]
        else:
            Lmin = 0.0
        if ndim == 1:
            neff = np.sqrt(2*np.abs(np.log(np.sqrt(2*np.pi)*np.std(XToy[yToy==0],ddof=1)*Lmin)))
        else:
            neff = np.sqrt(2*np.abs(np.log(np.sqrt(2*np.pi)*np.linalg.det(np.cov(XToy[yToy==0],rowvar=False))*Lmin)))
        if show_plots:
            plt.hist(likelihoods_toy,bins=cuts, histtype='step', density=False,alpha=0.5, label='Toys')
            plt.hist(likelihoods_cr, bins=cuts, histtype='step', density=False,alpha=0.5, label='Calibration Data')
            # plt.ylim()
            plt.axvline(Lmin,label=r'$\mathcal{L}_{\text{min}}$')
            plt.yscale('log')
            plt.xlabel(r'$\mathcal{L}_{\text{c}}$')
            plt.ylabel('Events')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/prob_calibration_events.pdf')
            plt.show()    

            plt.hist(likelihoods_toy,bins=cuts, histtype='step', density=True,alpha=0.5, label='Toys')
            plt.hist(likelihoods_cr, bins=cuts, histtype='step', density=True,alpha=0.5, label='Calibration Data')
            # plt.ylim()
            plt.axvline(Lmin,label=r'$\mathcal{L}_{\text{min}}$')
            plt.yscale('log')
            plt.xlabel(r'$\mathcal{L}_{\text{c}}$')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/prob_calibration.pdf')
            plt.show()            
            # print(np.std(XToy[yToy==0],ddof=1))
            # print(Lmin,neff)
            # # other_Lmin = np.exp(-(1**2)/2)/(np.sqrt(2*np.pi)*np.std(XToy[yToy==0],ddof=1))#
            # # print('Other Lmin:',other_Lmin,np.sqrt(2*np.abs(np.log(np.sqrt(2*np.pi)*np.std(XToy[yToy==0],ddof=1)*other_Lmin))))
            # plt.scatter(vals,fractions_toy,label='Toy')        
            # plt.scatter(vals,fractions,label='True')        
            # plt.axvline(Lmin,label=r'$\mathcal{L}_{\text{min}}$')
            # # plt.axhline(np.max((accumulated_fractions_toy-accumulated_fractions)))#,label=r'$\mathcal{L}_{\text{min}}$')
            # plt.xlabel(r'$\mathcal{L}_{\text{c}}$')
            # # plt.ylabel(r'|Data-Toy| Accumulated fraction')
            # plt.ylabel(r'Hypersurface fraction of events')
            # plt.legend(loc='best')
            # plt.tight_layout()
            # plt.savefig(f'{self.results_folder}/distribution_comparison.pdf')

            plt.show() 

        if show_plots:
            # print(np.std(XToy[yToy==0],ddof=1))
            print(Lmin,neff)
            # other_Lmin = np.exp(-(1**2)/2)/(np.sqrt(2*np.pi)*np.std(XToy[yToy==0],ddof=1))#
            # print('Other Lmin:',other_Lmin,np.sqrt(2*np.abs(np.log(np.sqrt(2*np.pi)*np.std(XToy[yToy==0],ddof=1)*other_Lmin))))
            plt.scatter(vals,(accumulated_fractions_toy-accumulated_fractions))        
            plt.axvline(Lmin,label=r'$\mathcal{L}_{\text{min}}$')
            plt.axhline(np.max((accumulated_fractions_toy-accumulated_fractions)))#,label=r'$\mathcal{L}_{\text{min}}$')
            plt.xlabel(r'$\mathcal{L}_{\text{c}}$')
            # plt.ylabel(r'|Data-Toy| Accumulated fraction')
            plt.ylabel(r'$f(\mathcal{L}_{\text{c}}|\theta_{\text{b}}) - f(\mathcal{L}_{\text{c}}|\text{true b})$')
            plt.legend(loc='best')
            plt.tight_layout()
            # plt.savefig(f'{self.results_folder}/distribution_comparison.pdf')

            plt.show()
        if show_plots:
            # print(np.std(XToy[yToy==0],ddof=1))
            print(Lmin,neff)
            # other_Lmin = np.exp(-(1**2)/2)/(np.sqrt(2*np.pi)*np.std(XToy[yToy==0],ddof=1))#
            # print('Other Lmin:',other_Lmin,np.sqrt(2*np.abs(np.log(np.sqrt(2*np.pi)*np.std(XToy[yToy==0],ddof=1)*other_Lmin))))
            plt.scatter(vals,np.abs(accumulated_fractions_toy-accumulated_fractions))        
            plt.axvline(Lmin,label=r'$\mathcal{L}_{\text{min}}$')
            plt.axhline(np.max(np.abs(accumulated_fractions_toy-accumulated_fractions)))#,label=r'$\mathcal{L}_{\text{min}}$')
            plt.xlabel(r'$\mathcal{L}_{\text{c}}$')
            # plt.ylabel(r'|Data-Toy| Accumulated fraction')
            plt.ylabel(r'$|f(\mathcal{L}_{\text{c}}|\theta_{\text{b}}) - f(\mathcal{L}_{\text{c}}|\text{true b})|$')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/distribution_comparison.pdf')

            plt.show()

        eff_bkg = np.mean(likelihoods_toy>Lmin)
        eff_cr = np.mean(likelihoods_cr>Lmin)
        eff_sig = np.mean(likelihoods_signal_toy>Lmin)

        if ndim == 1:
            c = cost.ExtendedUnbinnedNLL(X[likelihoods>Lmin], self.density_bias_with_cut)
        else:
            c = cost.ExtendedUnbinnedNLL(X[likelihoods>Lmin].T, self.density_bias_with_cut)

        m = Minuit(c, mu=1, Bfit=B,eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)
        #m.limits["mu"] = (0, None)
        ### I could add penalty terms here
        m.fixed["eff_bkg"] = True
        m.fixed["eff_sig"] = True
        m.fixed["eff_cr"] = True
        m.migrad()
        m.minos()

        central_value_biased_corrected = m.values['mu']
        error_biased_corrected = m.errors['mu']
        merror_biased_corrected = [float(repr(m.merrors['mu']).split('lower')[1].split('=')[1].split('upper')[0]),float(repr(m.merrors['mu']).split('upper')[1].split('=')[1].split('is_valid')[0])]  
        Bfit_val = m.values['Bfit']
        Bfit_error = m.errors['Bfit']

        if ndim == 1:
            norm1 = self.density_bias_with_cut(X[likelihoods>Lmin],S,B,eff_bkg, eff_sig, eff_cr)[0]
            norm2 = self.density_bias_with_cut(X[likelihoods>Lmin],*m.values)[0]
            llr_biased_corrected = -2*np.sum(np.log(self.density_bias_with_cut(X[likelihoods>Lmin],S,B,eff_bkg, eff_sig, eff_cr)[1]/norm1)-np.log(self.density_bias_with_cut(X[likelihoods>Lmin],*m.values)[1]/norm2))
        else:
            norm1 = self.density_bias_with_cut(X[likelihoods>Lmin].T,S,B,eff_bkg, eff_sig, eff_cr)[0]
            norm2 = self.density_bias_with_cut(X[likelihoods>Lmin].T,*m.values)[0]
            llr_biased_corrected = -2*np.sum(np.log(self.density_bias_with_cut(X[likelihoods>Lmin].T,S,B,eff_bkg, eff_sig, eff_cr)[1]/norm1)-np.log(self.density_bias_with_cut(X[likelihoods>Lmin].T,*m.values)[1]/norm2))


        if show_plots:
            grid_biased_corrected, profile_biased_corrected, bool_biased_corrected = m.mnprofile('mu')            

        if show_plots and ndim == 1:
            print(Lmin,eff_bkg,eff_cr,eff_sig)
            yields, bins, edges = plt.hist(X[likelihoods>Lmin],density=density_plot,histtype='step',bins=100,label='Data');
            vals = 0.5*(bins[1:]+bins[:-1])
            bin_size = bins[1]-bins[0]            
            if density_plot:
                plt.plot(vals,self.density_bias_with_cut(vals,mu=m.values['mu'], Bfit=m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[1]/self.density_bias_with_cut(vals,mu=m.values['mu'], Bfit=m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[0],label=f'$\mu_s$={central_value_biased_corrected/S:.3f} +/- {error_biased_corrected/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}')
            else:
                plt.plot(vals,bin_size*self.density_bias_with_cut(vals,mu=m.values['mu'], Bfit=m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[1],label=f'$\mu_s$={central_value_biased_corrected/S:.3f} +/- {error_biased_corrected/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}')
            plt.legend(loc='best')
            plt.ylabel('Events')
            plt.xlabel(r'$x$')
            plt.title(f'Biased + MRL fit, $\lambda_b ={round(B)}$, $\lambda_s={round(S)}$')                        
            plt.xlim(-4,4)
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/fit_biased_corrected.pdf')
            plt.show()    

        if show_plots and ndim == 2:

            yields, binsx, binsy, edges = plt.hist2d(X[likelihoods>Lmin][:,0],X[likelihoods>Lmin][:,1],density=density_plot,bins=[100,100],label='Data');
            valsx = 0.5*(binsx[1:]+binsx[:-1])
            valsy = 0.5*(binsy[1:]+binsy[:-1])

            bin_size_x = binsx[1]-binsx[0]
            bin_size_y = binsy[1]-binsy[0]

            # valsx_contour, valsy_contour = np.meshgrid(valsx,valsy)
            Z = np.zeros((len(valsy),len(valsx)))
            for i in range(len(valsy)):
                for j in range(len(valsx)):
                    val = np.array([valsx[j],valsy[i]]).reshape(-1,1)
                    # print(i,j,val,density(val,m.values['mu'],m.values['Bfit']))
                    if density_plot:
                        Z[i,j] = (self.density_bias_with_cut(val,m.values['mu'],m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[1]/self.density_bias_with_cut(val,m.values['mu'],m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[0])
                    else:
                        Z[i,j] = (bin_size_x*bin_size_y* self.density_bias_with_cut(val,m.values['mu'],m.values['Bfit'],eff_bkg=eff_bkg, eff_sig=eff_sig, eff_cr=eff_cr)[1])

            plt.contour(valsx,valsy,Z,levels=10,label=f'$\mu_s$={central_value_biased_corrected/S:.3f} +/- {error_biased_corrected/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}',cmap='plasma')
            # plt.colorbar()
            # plt.legend(loc='best')
            plt.text(0.05, 0.95, f'$\mu_s$={central_value_biased_corrected/S:.3f} +/- {error_biased_corrected/S:.3f}\n $\mu_b$={Bfit_val/B:.3f} +/- {Bfit_error/B:.3f}', transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.ylabel(r'$x_2$')
            plt.xlabel(r'$x_1$')
            plt.title(f'Biased + MRL fit, $\lambda_b ={round(B)}$, $\lambda_s={round(S)}$')
            plt.xlim(0,1)
            plt.ylim(0,1)
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/fit_biased_corrected.pdf')

            plt.show()   

        if show_plots:
            # print(grid_unbiased/S, 2*(profile_unbiased-np.min(profile_unbiased)))
            # print(grid_biased/S, 2*(profile_biased-np.min(profile_biased)))
            # print(grid_biased_corrected/S, 2*(profile_biased-np.min(profile_biased)))
            plt.plot(grid_unbiased/S, (profile_unbiased-np.min(profile_unbiased)),color='darkred',label='Unbiased')
            plt.plot(grid_biased/S, (profile_biased-np.min(profile_biased)),color='mediumblue',label='Biased')
            plt.plot(grid_biased_corrected/S, (profile_biased_corrected-np.min(profile_biased_corrected)),color='darkgreen',label='Biased + MRL')
            plt.xlabel("Signal Strength", fontsize=22)
            plt.ylabel(r"- 2 $\Delta$ ln($\mathcal{L}$)", fontsize=22)
            plt.axvline(1.0, color='red', linestyle='dashed', label='True signal strength')
            plt.legend(loc="upper center")
            plt.title(f'$\lambda_b ={round(B)}$, $\lambda_s={round(S)}$')
            plt.ylim([0, 4])
            plt.axhline(1.0,color='red')
            # plt.axvline(central_value_unbiased/S-error_unbiased/S,color='darkred',linestyle='dashed')
            # plt.axvline(central_value_unbiased/S+error_unbiased/S,color='darkred',linestyle='dashed')

            # plt.axvline(central_value_biased/S-error_biased/S,color='mediumblue',linestyle='dashed')
            # plt.axvline(central_value_biased/S+error_biased/S,color='mediumblue',linestyle='dashed')
            
            # plt.axvline(central_value_biased_corrected/S-error_biased_corrected/S,color='darkgreen',linestyle='dashed')
            # plt.axvline(central_value_biased_corrected/S+error_biased_corrected/S,color='darkgreen',linestyle='dashed')                        
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/profile_likelihood.pdf')
            plt.show()

    

        return [central_value_unbiased, error_unbiased, central_value_biased, error_biased, central_value_biased_corrected, error_biased_corrected], [merror_unbiased,merror_biased,merror_biased_corrected], [llr_unbiased,llr_biased,llr_biased_corrected], [Lmin,neff]
    
    def run_experiments(self,n_experiments,S,B,CRsize,Toysize,seeds,eps=0.01,density_plot=True):
        results = []
        for i in range(n_experiments):
            print(f'Running experiment {i+1}/{n_experiments} with seed {seeds[i]}')
            res = self.experiment(S=S,B=B,CRsize=CRsize,Toysize=Toysize,seed=seeds[i],eps=eps,show_plots=False,ndim=self.ndim,density_plot=density_plot)
            results.append(res)
        return results    
    def run_one_experiment_save_plots(self,S,B,CRsize,Toysize,seed,eps=0.01,density_plot=True):
        return self.experiment(S=S,B=B,CRsize=CRsize,Toysize=Toysize,seed=seed,eps=eps,show_plots=True,ndim=self.ndim,density_plot=density_plot)
    
    def run_Sinj_scan(self,Svals,n_experiments,B,CRsize,Toysize,seeds,eps=0.01,density_plot=True,y_coordinates=None,y_coordinates_2=None):
        try:
            Svals_test = np.load(f'{self.results_folder}/Svals.npy')
            all_results = np.load(f'{self.results_folder}/all_results.npy')
            if np.allclose(Svals_test,np.array(Svals)) and all_results.shape[0] == len(Svals) and all_results.shape[1] == n_experiments and all_results.shape[2] == 9:
                loaded = True
            else:
                loaded = False
        except:
            loaded = False
        if loaded == False:
            all_results = np.zeros((len(Svals),n_experiments,9))
        for nS, Sinj in enumerate(Svals):
            # results = np.zeros((n_experiments,6))
            if loaded == False:
                neffs = np.zeros(n_experiments)
                for nexp in range(n_experiments):
                    temp_result = self.experiment(S=Sinj,B=B,CRsize=CRsize,Toysize=Toysize,seed=nS+nexp,eps=eps,show_plots=False,ndim=self.ndim,density_plot=density_plot)
                    all_results[nS,nexp,:6] = temp_result[0]
                    all_results[nS,nexp,6:] = temp_result[2]
                    neffs[nexp] = temp_result[3][1]
                    if (nexp+1)%100 == 0:
                        print(nexp+1)
            if nS == 0:
                plt.errorbar(Sinj+1*norm(0,0.1).rvs(n_experiments),all_results[nS,:,0]/Sinj, yerr=all_results[nS,:,1]/Sinj, color='darkred', fmt='o', alpha=1,label='Unbiased')
                plt.errorbar(Sinj+1*norm(0,0.1).rvs(n_experiments),all_results[nS,:,2]/Sinj, yerr=all_results[nS,:,3]/Sinj,  color='mediumblue',fmt='x', alpha=1,label='Biased')
                plt.errorbar(Sinj+1*norm(0,0.1).rvs(n_experiments),all_results[nS,:,4]/Sinj, yerr=all_results[nS,:,5]/Sinj, color='darkgreen',fmt='+', alpha=1,label='Biased + MRL')
                plt.axhline(1.0,color='k',ls='--')
            else:
                plt.errorbar(Sinj+1*norm(0,0.1).rvs(n_experiments),all_results[nS,:,0]/Sinj, yerr=all_results[nS,:,1]/Sinj, color='darkred',fmt='o', alpha=1)
                plt.errorbar(Sinj+1*norm(0,0.1).rvs(n_experiments),all_results[nS,:,2]/Sinj, yerr=all_results[nS,:,3]/Sinj, color='mediumblue',fmt='x', alpha=1)
                plt.errorbar(Sinj+1*norm(0,0.1).rvs(n_experiments),all_results[nS,:,4]/Sinj, yerr=all_results[nS,:,5]/Sinj, color='darkgreen',fmt='+', alpha=1)
                # plt.axhline(1.0,color='k',ls='--')
            # print(neffs)
            # print(np.mean(neffs),np.std(neffs,ddof=1))

            if y_coordinates == None:
                plt.annotate(
                    "$\mu=%8.2f \pm %8.2f$" % (np.mean(all_results[nS,:,0])/Sinj, np.mean(all_results[nS,:,1])/Sinj),
                    xy=(Sinj, 1.0),
                    xytext=(Sinj,  np.mean(all_results[nS,:,0])/Sinj-2*np.mean(all_results[nS,:,1])/Sinj),#np.mean(all_results[nS,:,0])/Sinj),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    # textcoords='figure fraction',
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='darkred',bbox=dict(facecolor='white', alpha=0.75) 
                )
                plt.annotate(
                    "$\mu=%8.2f \pm %8.2f$" % (np.mean(all_results[nS,:,2])/Sinj, np.mean(all_results[nS,:,3])/Sinj),
                    xy=(Sinj,  1.0),
                    xytext=(Sinj, np.mean(all_results[nS,:,2])/Sinj+2*np.mean(all_results[nS,:,3])/Sinj),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='mediumblue',bbox=dict(facecolor='white', alpha=0.75) 
                )    
                plt.annotate(
                    "$\mu=%8.2f \pm %8.2f$" % (np.mean(all_results[nS,:,4])/Sinj, np.mean(all_results[nS,:,5])/Sinj),
                    xy=(Sinj, 1.0),
                    xytext=(Sinj,np.mean(all_results[nS,:,4])/Sinj),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='darkgreen',bbox=dict(facecolor='white', alpha=0.75) 
                )
            else:
                plt.annotate(
                    "$\mu=%8.2f \pm %8.2f$" % (np.mean(all_results[nS,:,0])/Sinj, np.mean(all_results[nS,:,1])/Sinj),
                    xy=(Sinj, 1.0),
                    xytext=(Sinj,y_coordinates[nS][0]),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    # textcoords='figure fraction',
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='darkred',bbox=dict(facecolor='white', alpha=0.75) 
                )
                plt.annotate(
                    "$\mu=%8.2f \pm %8.2f$" % (np.mean(all_results[nS,:,2])/Sinj, np.mean(all_results[nS,:,3])/Sinj),
                    xy=(Sinj,  1.0),
                    xytext=(Sinj, y_coordinates[nS][1]),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='mediumblue',bbox=dict(facecolor='white', alpha=0.75) 
                )    
                plt.annotate(
                    "$\mu=%8.2f \pm %8.2f$" % (np.mean(all_results[nS,:,4])/Sinj, np.mean(all_results[nS,:,5])/Sinj),
                    xy=(Sinj, 1.0),
                    xytext=(Sinj,y_coordinates[nS][2]),
                    # xytext=(Sinj, 0.6),#np.mean(all_results[nS,:,4])/Sinj),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='darkgreen',bbox=dict(facecolor='white', alpha=0.75) 
                )
        plt.xlabel('True signal events')
        plt.ylabel('Fitted signal strength')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/signal_injection_study.png')
        plt.savefig(f'{self.results_folder}/signal_injection_study.pdf')
        plt.show()
        # if np.min(Svals) > 0:
        #     plt.xscale('log')
        #     plt.yscale('log')  
        #     plt.tight_layout()      
        #     plt.savefig(f'{self.results_folder}/signal_injection_study_log.png')
        #     plt.savefig(f'{self.results_folder}/signal_injection_study_log.pdf')
        if loaded == False:
            np.save(f'{self.results_folder}/Svals.npy',Svals)
            np.save(f'{self.results_folder}/all_results.npy',all_results)


        for nS, Sinj in enumerate(Svals):
            # results = np.zeros((n_experiments,6))
            # print(Sinj,all_results[nS,:,4],all_results[nS,:,5])
            # print(np.abs(all_results[nS,:,4]-Sinj[np.newaxis])<all_results[nS,:,5])
            p1 = np.mean(np.abs(all_results[nS,:,0]-Sinj[np.newaxis])<all_results[nS,:,1])
            p2 = np.mean(np.abs(all_results[nS,:,2]-Sinj[np.newaxis])<all_results[nS,:,3])
            p3 = np.mean(np.abs(all_results[nS,:,4]-Sinj[np.newaxis])<all_results[nS,:,5])


            if nS == 0:
                plt.errorbar(Sinj,p1, yerr=np.sqrt(p1*(1-p1)/n_experiments), color='darkred', fmt='o', alpha=1,label='Unbiased')
                plt.errorbar(Sinj,p2, yerr=np.sqrt(p2*(1-p2)/n_experiments),  color='mediumblue',fmt='x', alpha=1,label='Biased')
                plt.errorbar(Sinj,p3, yerr=np.sqrt(p3*(1-p3)/n_experiments), color='darkgreen',fmt='+', alpha=1,label='Biased + MRL')
                plt.axhline(0.68,color='k',ls='--')
            else:
                plt.errorbar(Sinj,p1, yerr=np.sqrt(p1*(1-p1)/n_experiments), color='darkred', fmt='o', alpha=1)
                plt.errorbar(Sinj,p2, yerr=np.sqrt(p2*(1-p2)/n_experiments),  color='mediumblue',fmt='x', alpha=1)
                plt.errorbar(Sinj,p3, yerr=np.sqrt(p3*(1-p3)/n_experiments), color='darkgreen',fmt='+', alpha=1)
                # plt.axhline(1.0,color='k',ls='--')
            # print(neffs)
            # print(np.mean(neffs),np.std(neffs,ddof=1))

            if y_coordinates_2 == None:
                plt.annotate(
                    "$p=%8.2f \pm %8.2f$" % (p1,np.sqrt(p1*(1-p1)/n_experiments)),
                    xy=(Sinj, p1),
                    # xytext=(Sinj,  np.mean(np.abs(all_results[nS,:,0]-Sinj)<all_results[nS,:,1])),#np.mean(all_results[nS,:,0])/Sinj),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    # textcoords='figure fraction',
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='darkred',bbox=dict(facecolor='white', alpha=0.75) 
                )
                plt.annotate(
                    "$p=%8.2f \pm %8.2f$" % (p2,np.sqrt(p2*(1-p2)/n_experiments)),
                    xy=(Sinj, p2),
                    # xytext=(Sinj, np.max([np.mean(all_results[nS,:,2])/Sinj,2.2])),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='mediumblue',bbox=dict(facecolor='white', alpha=0.75) 
                )    
                plt.annotate(
                    "$p=%8.2f \pm %8.2f$" % (p3,np.sqrt(p3*(1-p3)/n_experiments)),
                    xy=(Sinj, p3),
                    # xytext=(Sinj,np.mean(all_results[nS,:,4])/Sinj),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='darkgreen',bbox=dict(facecolor='white', alpha=0.75) 
                )
            else:
                plt.annotate(
                    "$p=%8.2f \pm %8.2f$" % (p1,np.sqrt(p1*(1-p1)/n_experiments)),
                    xy=(Sinj, 0.68),
                    xytext=(Sinj,y_coordinates_2[nS][0]),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    # textcoords='figure fraction',
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='darkred',bbox=dict(facecolor='white', alpha=0.75) 
                )
                plt.annotate(
                    "$p=%8.2f \pm %8.2f$" % (p2,np.sqrt(p2*(1-p2)/n_experiments)),
                    xy=(Sinj,  0.68),
                    xytext=(Sinj, y_coordinates_2[nS][1]),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='mediumblue',bbox=dict(facecolor='white', alpha=0.75) 
                )    
                plt.annotate(
                    "$p=%8.2f \pm %8.2f$" % (p3,np.sqrt(p3*(1-p3)/n_experiments)),
                    xy=(Sinj, 0.68),
                    xytext=(Sinj,y_coordinates_2[nS][2]),
                    # xytext=(Sinj, 0.6),#np.mean(all_results[nS,:,4])/Sinj),
                    # arrowprops=dict(facecolor="black", shrink=0.05),
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=10,
                    color='darkgreen',bbox=dict(facecolor='white', alpha=0.75) 
                )
        plt.xlabel('True signal events')
        plt.ylabel('Coverage test')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/signal_injection_study_coverage.png')
        plt.savefig(f'{self.results_folder}/signal_injection_study_coverage.pdf')
        plt.show()

        for nS, Sinj in enumerate(Svals):
            # results = np.zeros((n_experiments,6))
            # print(Sinj,all_results[nS,:,4],all_results[nS,:,5])
            # print(np.abs(all_results[nS,:,4]-Sinj[np.newaxis])<all_results[nS,:,5])
            p1 = np.mean(np.abs(all_results[nS,:,0]-Sinj[np.newaxis])<all_results[nS,:,1])
            f1 = np.mean(all_results[nS,:,0]<0)

            p2 = np.mean(np.abs(all_results[nS,:,2]-Sinj[np.newaxis])<all_results[nS,:,3])
            f2 = np.mean(all_results[nS,:,2]<0)

            p3 = np.mean(np.abs(all_results[nS,:,4]-Sinj[np.newaxis])<all_results[nS,:,5])
            f3 = np.mean(all_results[nS,:,4]<0)


            if nS == 0:
                plt.errorbar(f1,p1,xerr=np.sqrt(f1*(1-f1)/n_experiments), yerr=np.sqrt(p1*(1-p1)/n_experiments), color='darkred', fmt='o', alpha=1,label='Unbiased')
                plt.errorbar(f2,p2,xerr=np.sqrt(f2*(1-f2)/n_experiments), yerr=np.sqrt(p2*(1-p2)/n_experiments),  color='mediumblue',fmt='x', alpha=1,label='Biased')
                plt.errorbar(f3,p3,xerr=np.sqrt(f3*(1-f3)/n_experiments), yerr=np.sqrt(p3*(1-p3)/n_experiments), color='darkgreen',fmt='+', alpha=1,label='Biased + MRL')
                plt.axhline(0.68,color='k',ls='--')
                plt.axvline(0.0,color='k',ls='--')
            else:
                plt.errorbar(f1,p1,xerr=np.sqrt(f1*(1-f1)/n_experiments), yerr=np.sqrt(p1*(1-p1)/n_experiments), color='darkred', fmt='o', alpha=1)
                plt.errorbar(f2,p2,xerr=np.sqrt(f2*(1-f2)/n_experiments), yerr=np.sqrt(p2*(1-p2)/n_experiments),  color='mediumblue',fmt='x', alpha=1)
                plt.errorbar(f3,p3,xerr=np.sqrt(f3*(1-f3)/n_experiments), yerr=np.sqrt(p3*(1-p3)/n_experiments), color='darkgreen',fmt='+', alpha=1)
                # plt.axhline(1.0,color='k',ls='--')
            # print(neffs)
            # print(np.mean(neffs),np.std(neffs,ddof=1))

        plt.xlabel('Fraction of negative $\hat{S}$')
        plt.ylabel('Coverage test')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f'{self.results_folder}/coverage_negative_fraction.png')
        plt.savefig(f'{self.results_folder}/coverage_negative_fraction.pdf')
        plt.show()


        for nS, Sinj in enumerate(Svals):
            # results = np.zeros((n_experiments,6))
            # print(Sinj,all_results[nS,:,4],all_results[nS,:,5])
            # print(np.abs(all_results[nS,:,4]-Sinj[np.newaxis])<all_results[nS,:,5])
            plt.plot(np.linspace(-2,2,100),st.norm().pdf(np.linspace(-2,2,100)),color='black',label='Standard Normal')
            plt.hist((all_results[nS,:,0]-Sinj[np.newaxis])/all_results[nS,:,1], color='darkred', histtype='step', alpha=1,label='Unbiased',density=True)
            plt.hist((all_results[nS,:,2]-Sinj[np.newaxis])/all_results[nS,:,3],  color='mediumblue',histtype='step', alpha=1,label='Biased',density=True)
            plt.hist((all_results[nS,:,4]-Sinj[np.newaxis])/all_results[nS,:,5], color='darkgreen',histtype='step', alpha=1,label='Biased + MRL',density=True)
            plt.axvline(0.0,color='k',ls='--')

            plt.xlabel('Pull')
            plt.ylabel('Seudo-experiment')
            # plt.xscale('log')
            # plt.yscale('log')
            plt.legend(loc='best')
            plt.title(r'$\lambda_s$='+str(Sinj))
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/pulls_{nS}.png')
            plt.savefig(f'{self.results_folder}/pulls_{nS}.pdf')
            plt.show()

            plt.hist(all_results[nS,:,0], color='darkred', histtype='step', alpha=1,label='Unbiased',density=True)
            plt.hist(all_results[nS,:,2],  color='mediumblue',histtype='step', alpha=1,label='Biased',density=True)
            plt.hist(all_results[nS,:,4], color='darkgreen',histtype='step', alpha=1,label='Biased + MRL',density=True)
            plt.axvline(0.0,color='k',ls='--')

            plt.xlabel('$\hat{S}$')
            plt.ylabel('Seudo-experiment')
            plt.axvline(Sinj,color='black',linestyle='dashed')
            # plt.xscale('log')
            # plt.yscale('log')
            plt.legend(loc='best')
            plt.title(r'$\lambda_s$='+str(Sinj))
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/pulls_{nS}.png')
            plt.savefig(f'{self.results_folder}/pulls_{nS}.pdf')
            plt.show()            
        
            # plt.plot(np.linspace(-2,2,100),st.norm().pdf(np.linspace(-2,2,100)),color='black',label='Standard Normal')
            plt.hist(all_results[nS,:,6], color='darkred', histtype='step', alpha=1,label='Unbiased',density=True)
            # plt.axvline(0.0,color='k',ls='--')

            plt.xlabel('Test statistic')
            plt.ylabel('Seudo-experiment')
            # plt.xscale('log')
            # plt.yscale('log')
            plt.legend(loc='best')
            plt.title(r'$\lambda_s$='+str(Sinj))
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/llr_unbiased_{nS}.png')
            plt.savefig(f'{self.results_folder}/llr_unbiased_{nS}.pdf')
            plt.show()

            plt.hist(all_results[nS,:,7],  color='mediumblue',histtype='step', alpha=1,label='Biased',density=True)
            # plt.axvline(0.0,color='k',ls='--')

            plt.xlabel('Test statistic')
            plt.ylabel('Seudo-experiment')
            # plt.xscale('log')
            # plt.yscale('log')
            plt.legend(loc='best')
            plt.title(r'$\lambda_s$='+str(Sinj))
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/llr_biased_{nS}.png')
            plt.savefig(f'{self.results_folder}/llr_biased_{nS}.pdf')
            plt.show()

            plt.hist(all_results[nS,:,8], color='darkgreen',histtype='step', alpha=1,label='Biased + MRL',density=True)
            # plt.axvline(0.0,color='k',ls='--')

            plt.xlabel('Test statistic')
            plt.ylabel('Seudo-experiment')
            # plt.xscale('log')
            # plt.yscale('log')
            plt.legend(loc='best')
            plt.title(r'$\lambda_s$='+str(Sinj))
            plt.tight_layout()
            plt.savefig(f'{self.results_folder}/llr_biased_corrected_{nS}.png')
            plt.savefig(f'{self.results_folder}/llr_biased_corrected_{nS}.pdf')
            plt.show()