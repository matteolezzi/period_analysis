import numpy as np
import matplotlib.pyplot as plt
from sigfig import round
from astropy.timeseries import LombScargle

# legge la lista delle sorgenti (solo il nome del file della curva di luce)
lcname = np.loadtxt("lista_sorgenti.txt", dtype=str, usecols=(2), unpack=True)

# figura riassuntiva con tutte le curve di luce
fig, axs = plt.subplots(6, 5, figsize=(15, 9))
axs = axs.flatten()

pmin = 1  # periodo minimo di ricerca [ore]
pmax = np.array([50,18,40,18,18,18,30,18,18,40,12,50,18,50,20,20,30,25,20,120,20,20,120,30,25,20,20,30,20,100])
Ncicli = 100  # numero di ricampionamenti per il bootstrap


def FoldLightCurve2(time, flux, error, period, nbins=10, t0=0):
    # fase di ogni punto, compresa in [0,1)
    epoch = np.floor((time - t0) / period)
    phases = (time - t0) / period - epoch

    # ordina per fase
    sorted_Index_phases = np.argsort(phases)
    sorted_phases = phases[sorted_Index_phases]
    sorted_flux = flux[sorted_Index_phases]
    sorted_error = error[sorted_Index_phases]

    deltaphase = 1 / float(nbins)
    phase = np.zeros(nbins)
    profile = np.zeros(nbins)
    proferr = np.zeros(nbins)

    # media di flusso ed errore in ogni bin di fase
    for ibin in range(nbins):
        phase_bin = deltaphase * ibin
        phase[ibin] = phase_bin
        index = np.where((sorted_phases >= phase_bin) & (sorted_phases < phase_bin + deltaphase))
        if len(index[0]) > 0:
            profile[ibin] = np.mean(sorted_flux[index])
            proferr[ibin] = np.mean(sorted_error[index])
        else:
            # bin vuoto
            phase[ibin] = np.nan
            profile[ibin] = np.nan
            proferr[ibin] = np.nan

    # toglie i bin vuoti
    index = np.where(np.isfinite(profile))
    return phase[index], profile[index], proferr[index]


for i in range(len(lcname)):
    mjd, mag, errmag = np.loadtxt(lcname[i], usecols=(0, 1, 2), unpack=True)
    mjd = mjd - mjd[0]  # tempo iniziale a 0
    time = mjd * 24     # da giorni a ore

    # curva di luce nella figura riassuntiva
    axs[i].plot(time, mag)
    axs[i].invert_yaxis()
    axs[i].set_title('Variabile ' + str(i + 1))
    axs[i].set_xlabel('Time [hours]')
    axs[i].set_ylabel('magnitude')

    # periodogramma e periodo più probabile
    frequency1, power1 = LombScargle(time, mag).autopower(minimum_frequency=1 / pmax[i], maximum_frequency=1 / pmin,
                                                          samples_per_peak=100)
    period1 = 1 / frequency1
    max_index = np.argmax(power1)
    max_period = period1[max_index]
    max_period_day = max_period / 24

    fig1, axs1 = plt.subplots(3, 1, figsize=(15, 9))

    # errore sul periodo con bootstrap:
    # si ricampionano i dati originali con reinserimento e si ricalcola il periodo
    pmax_err = np.empty(Ncicli)
    for j in range(Ncicli):
        idx = np.random.choice(len(time), size=len(time), replace=True)
        frequency, power = LombScargle(time[idx], mag[idx]).autopower(minimum_frequency=1 / pmax[i],
                                                                      maximum_frequency=1 / pmin,
                                                                      samples_per_peak=100)
        pmax_err[j] = 1 / frequency[np.argmax(power)]
    sigma = np.std(pmax_err)
    sigma_day = sigma / 24

    # curva di luce con errori
    axs1[0].errorbar(time, mag, ls='None', yerr=errmag, marker="o", markersize=3)
    axs1[0].invert_yaxis()
    axs1[0].set_title('Variabile ' + str(i + 1), fontsize=30)
    axs1[0].set_xlabel('Time [hours]', fontsize=30)
    axs1[0].set_ylabel('magnitude', fontsize=20)

    # periodogramma con il periodo e il suo errore in legenda
    axs1[1].plot(period1, power1, label="P = " + str(round(max_period, sigma, cutoff=35)) + " hours = " + str(round(max_period_day, sigma_day, cutoff=35)) + " days")
    axs1[1].axvline(x=max_period, color='red')
    axs1[1].set_title('Periodogramma ' + str(i + 1), fontsize=30)
    axs1[1].set_xlabel('Period [hours]', fontsize=30)
    axs1[1].set_ylabel('Power', fontsize=20)
    axs1[1].legend(fontsize=20, loc=0)

    # curva ripiegata sul periodo, ripetuta su due cicli (fase 0-2)
    phase, profile, proferr = FoldLightCurve2(time, mag, errmag, max_period, nbins=40)
    concatenated_phase = np.concatenate([phase, phase + 1])
    concatenated_profile = np.concatenate([profile, profile])
    concatenated_proferr = np.concatenate([proferr, proferr])

    axs1[2].invert_yaxis()
    axs1[2].errorbar(concatenated_phase, concatenated_profile, ls='None', yerr=concatenated_proferr, marker="o", markersize=3)
    axs1[2].set_title('Folded light curve ' + str(i + 1), fontsize=30)
    axs1[2].set_xlabel('Phase', fontsize=30)
    axs1[2].set_ylabel('magnitude', fontsize=20)

    for ax in axs1:
        ax.tick_params(axis='x', which='major', labelsize=16)
        ax.tick_params(axis='y', which='major', labelsize=13)

fig.tight_layout()
plt.show()
