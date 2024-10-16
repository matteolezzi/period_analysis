import numpy as np
import matplotlib.pyplot as plt
from sigfig import round
from astropy.timeseries import LombScargle

#legge dati

ra, dec = np.loadtxt("lista_sorgenti.txt", usecols=(0,1), unpack=True)
lcname = np.loadtxt("lista_sorgenti.txt", dtype=str, usecols=(2), unpack=True)

#legge singola curva di luce e plotta

fig, axs = plt.subplots(6,5, figsize=(15, 9))  #da una figura ed un array

axs = axs.flatten() #per iterare su una dimensione
pmin = 1
pmax = np.array([50,18,40,18,18,18,30,18,18,40,12,50,18,50,20,20,30,25,20,120,20,20,120,30,25,20,20,30,20,100])

powermax = np.empty(30) #definisce array per descrivere il massimo del singolo periodogramma
Ncicli = 5
for i in range(len(lcname)):
     mjd, mag, errmag = np.loadtxt(lcname[i], usecols=(0, 1,2 ), unpack=True)
     mjd = mjd - mjd[0] #mette a 0 il tempo iniziale
     time = mjd * 24 #cambia in ore

     timefull = time.copy()
     magfull = mag.copy()
     errmagfull = errmag.copy()
    #plot

     axs[i].plot(time, mag)
     axs[i].invert_yaxis()
     axs[i].set_title('Variabile ' + str(i + 1))
     axs[i].set_xlabel('Time [hours]')
     axs[i].set_ylabel('magnitude')
     plt.tight_layout()

    #calcolo della frequenza e probabilità
     frequency1, power1 = LombScargle(time, mag).autopower(minimum_frequency=1 / pmax[i], maximum_frequency=1 / pmin,
                                                         samples_per_peak=100)
     period1 = 1 / frequency1
     periodfull = period1.copy()
     fig1, axs1 = plt.subplots(3, 1, figsize=(15, 9))
     axs1 = axs1.flatten() #per iterare su una dimensione
     max_index = np.argmax(power1)
     max_period = period1[max_index] #calcolo del periodo più probabile
     max_period_day= max_period/24
    #calcolo dell'errore
     pmax_err = np.empty(Ncicli)
     arr = np.arange(len(time))
     for j in range(Ncicli):
         for k in range(len(time)):
             random_index = np.random.choice(arr)
             time[k] = time[random_index]
             mag[k] = mag[random_index]
         frequency, power = LombScargle(time, mag).autopower(minimum_frequency=1 / pmax[i], maximum_frequency=1 / pmin,
                                                            samples_per_peak=100)
         period = 1 / frequency
         max_index_err = np.argmax(power)
         pmax_err[j] = period[max_index_err]
     #calcolo della deviazione standard per tutte le curve di luce
     sigma = np.std(pmax_err)
     sigma_day = sigma/24

     axs1[0].errorbar(timefull, magfull, ls = 'None' , yerr = errmagfull, marker = "o", markersize = 3)
     axs1[1].plot(period1, power1, label="P = " + str(round(max_period, sigma, cutoff =35)) + " hours = " + str(round(max_period_day, sigma_day, cutoff =35)) + " days")
     axs1[0].invert_yaxis()
     axs1[0].set_title('Variabile ' + str(i + 1),  fontsize = 30)
     axs1[0].set_xlabel('Time [hours]',  fontsize = 30)
     axs1[0].set_ylabel('magnitude',  fontsize = 20)
     axs1[1].set_title('Periodogramma ' + str(i + 1),  fontsize = 30)
     axs1[1].set_xlabel('Period [hours]', fontsize = 30)
     axs1[1].set_ylabel('Power',  fontsize = 20)  # aggiusta il layout
     axs1[0].tick_params(axis='x', which='major', labelsize=16)
     axs1[0].tick_params(axis='y', which='major', labelsize=13)
     axs1[1].tick_params(axis='x', which='major', labelsize=16)
     axs1[1].tick_params(axis='y', which='major', labelsize=13)
     # plt.show()
     axs1[1].axvline(x=max_period, color='red')
    #aggiunta tabella nella posizione migliore
     axs1[1].legend(fontsize = 20 ,loc = 0)


     def FoldLightCurve2(time, flux, error, period, nbins=10, t0=0, printout=False):

         # print('method: evend/odd spacing and binning')
         # Find epoch
         epoch = np.floor((time - t0) / period)
         # Find phase
         phases = (time - t0) / period - epoch

         # trova minimo bin in fase
         minphasebin = 9999
         for i in range(0, len(phases) - 1):
              bin = phases[i + 1] - phases[i]
              if (bin > 0 and bin < minphasebin):
                  minphasebin = bin

         # sort phases
         sorted_Index_phases = np.argsort(phases)
         sorted_phases = phases[sorted_Index_phases]
         sorted_flux = flux[sorted_Index_phases]
         sorted_error = error[sorted_Index_phases]

         # Settaggi iniziali per la fase
         phasemin = 0
         phasemax = 1
         deltaphase = (phasemax - phasemin) / float(nbins)

         # Se il numero di bin richiesto eccede, allora non ha senso. Lo limito
         nbinsold = nbins

         # Rebin output with Nbins!
         phase = np.zeros(nbins)
         profile = np.zeros(nbins)
         proferr = np.zeros(nbins)

         # Rebinning....
         couint = 0
         for ibin in range(0, nbins):
             phase_bin = phasemin + deltaphase * (ibin)
             phase[ibin] = phase_bin
             index = np.where((sorted_phases >= phase_bin) & (sorted_phases < phase_bin + deltaphase))
             couint = couint + len(index[0])
             # select values in bin
             flux_bin = sorted_flux[index]
             error_bin = sorted_error[index]
             if (len(index[0]) > 0):
                  sum = 0
                  for ifluxbin in flux_bin:
                   sum = sum + ifluxbin
                  sum = sum / float(len(flux_bin))
                  profile[ibin] = sum
                  sum = 0
                  for ifluxbin in error_bin:
                   sum = sum + ifluxbin
                  sum = sum / float(len(error_bin))
                  proferr[ibin] = sum
                  if (printout):
                   print(ibin, phase_bin, phase_bin + deltaphase, len(index[0]), profile[ibin], proferr[ibin])
             else:
                  # if no counts in bin, fill with nans
                  proferr[ibin] = np.nan
                  profile[ibin] = np.nan
                  phase[ibin] = np.nan

         # Removing nans in output!
         index = np.where(np.isfinite(profile))
         if (len(index[0]) > 0):
            profile = profile[index]
            proferr = proferr[index]
            phase = phase[index]

         return phase, profile, proferr

     phase, profile, proferr = FoldLightCurve2(timefull, magfull, errmagfull , max_period, nbins=40)

     concatenated_phase = np.concatenate([phase, phase + 1])
     concatenated_profile = np.concatenate([profile, profile])
     concatenated_proferr = np.concatenate([proferr, proferr])

     axs1[2].invert_yaxis()
     axs1[2].errorbar(concatenated_phase, concatenated_profile, ls='None', yerr=concatenated_proferr, marker="o", markersize=3)
     axs1[2].set_title('Folded light curve ' + str(i + 1), fontsize = 30)
     axs1[2].set_xlabel('Phase', fontsize = 30)
     axs1[2].set_ylabel('magnitude', fontsize = 20)
     axs1[2].tick_params(axis='x', which='major', labelsize=16)
     axs1[2].tick_params(axis='y', which='major', labelsize=13)


plt.tight_layout() # aggiusta il layout
plt.show()


