"""
Analisi di curve di luce di stelle variabili.

Per ogni sorgente:
  1. legge la curva di luce (tempo, magnitudine, errore);
  2. calcola il periodogramma di Lomb-Scargle e trova il periodo del picco massimo;
  3. stima l'errore sul periodo con un bootstrap;
  4. ripiega la curva di luce sul periodo trovato (curva in fase).
"""

import numpy as np
import matplotlib.pyplot as plt
from sigfig import round as sfround          # rinominata: non sovrascrive la round() di Python
from astropy.timeseries import LombScargle


# ----------------------------------------------------------------------------
# FUNZIONI
# ----------------------------------------------------------------------------

def fold_light_curve(time, mag, err, period, nbins=40, t0=0.0):
    """
    Ripiega la curva di luce su un dato periodo e ne fa la media in nbins bin di fase.

    Restituisce: fase (inizio di ogni bin), magnitudine media, errore sulla media.
    I bin vuoti vengono scartati.
    """
    # Fase di ogni punto: parte frazionaria di (t - t0)/P, quindi compresa in [0, 1)
    phases = ((time - t0) / period) % 1.0

    # Indice del bin a cui appartiene ogni punto (0 ... nbins-1)
    bin_idx = np.minimum((phases * nbins).astype(int), nbins - 1)

    phase_out, prof_out, err_out = [], [], []
    for b in range(nbins):
        in_bin = bin_idx == b
        n = in_bin.sum()
        if n == 0:
            continue                                     # bin vuoto: lo saltiamo
        phase_out.append(b / nbins)                      # fase di inizio bin
        prof_out.append(mag[in_bin].mean())              # magnitudine media nel bin
        # errore sulla media: propagazione degli errori dei singoli punti, sqrt(sum(e^2))/N
        err_out.append(np.sqrt(np.sum(err[in_bin] ** 2)) / n)

    return np.array(phase_out), np.array(prof_out), np.array(err_out)


def best_period(time, mag, err, fmin, fmax):
    """Restituisce il periodo (ore) del picco più alto del periodogramma, più il periodogramma."""
    freq, power = LombScargle(time, mag, err).autopower(
        minimum_frequency=fmin, maximum_frequency=fmax, samples_per_peak=100)
    period = 1.0 / freq
    return period[np.argmax(power)], period, power


def format_period(value, sigma):
    """Arrotonda il valore in base alla sua incertezza (con fallback se sigma è nullo)."""
    if sigma > 0:
        return str(sfround(value, sigma, cutoff=35))
    return f"{value:.4g}"


# ----------------------------------------------------------------------------
# PARAMETRI
# ----------------------------------------------------------------------------

rng = np.random.default_rng()      # generatore di numeri casuali per il bootstrap

pmin = 1                           # periodo minimo di ricerca [ore], uguale per tutte
# periodo massimo di ricerca [ore], uno per ogni sorgente
pmax = np.array([50, 18, 40, 18, 18, 18, 30, 18, 18, 40, 12, 50, 18, 50, 20,
                 20, 30, 25, 20, 120, 20, 20, 120, 30, 25, 20, 20, 30, 20, 100])

Ncicli = 100                       # numero di ricampionamenti bootstrap (5 era troppo poco)
nbins = 40                         # numero di bin di fase per la curva ripiegata

# ----------------------------------------------------------------------------
# LETTURA DELLA LISTA DELLE SORGENTI
# colonne: ra, dec, nome del file della curva di luce
# ----------------------------------------------------------------------------

ra, dec = np.loadtxt("lista_sorgenti.txt", usecols=(0, 1), unpack=True)   # non usate nell'analisi
lcname = np.loadtxt("lista_sorgenti.txt", dtype=str, usecols=(2), unpack=True)

# Figura riassuntiva: una curva di luce per ogni sorgente (griglia 6x5)
fig, axs = plt.subplots(6, 5, figsize=(15, 9))
axs = axs.flatten()

# ----------------------------------------------------------------------------
# CICLO SULLE SORGENTI
# ----------------------------------------------------------------------------

for i in range(len(lcname)):

    # --- 1. Lettura e preparazione dei dati --------------------------------
    mjd, mag, errmag = np.loadtxt(lcname[i], usecols=(0, 1, 2), unpack=True)
    mjd = mjd - mjd[0]             # il tempo parte da 0
    time = mjd * 24                # da giorni a ore

    # --- 2. Pannello della figura riassuntiva ------------------------------
    axs[i].plot(time, mag)
    axs[i].invert_yaxis()          # in astronomia magnitudine minore = più brillante
    axs[i].set_title('Variabile ' + str(i + 1))
    axs[i].set_xlabel('Time [hours]')
    axs[i].set_ylabel('magnitude')

    # --- 3. Periodogramma e periodo più probabile --------------------------
    max_period, period1, power1 = best_period(time, mag, errmag,
                                              fmin=1 / pmax[i], fmax=1 / pmin)
    max_period_day = max_period / 24

    # --- 4. Errore sul periodo con bootstrap -------------------------------
    # Si estraggono N punti CON reinserimento dai dati originali (che restano intatti),
    # si ricalcola il periodo e si ripete Ncicli volte.
    # La deviazione standard dei periodi ottenuti è l'incertezza.
    pmax_err = np.empty(Ncicli)
    for j in range(Ncicli):
        idx = rng.integers(0, len(time), size=len(time))     # indici casuali con reinserimento
        pmax_err[j], _, _ = best_period(time[idx], mag[idx], errmag[idx],
                                        fmin=1 / pmax[i], fmax=1 / pmin)
    sigma = np.std(pmax_err)       # errore in ore
    sigma_day = sigma / 24         # errore in giorni

    # --- 5. Figura di dettaglio: curva di luce, periodogramma, curva in fase
    fig1, axs1 = plt.subplots(3, 1, figsize=(15, 9))

    # 5a. Curva di luce con barre d'errore
    axs1[0].errorbar(time, mag, yerr=errmag, ls='None', marker='o', markersize=3)
    axs1[0].invert_yaxis()
    axs1[0].set_title('Variabile ' + str(i + 1), fontsize=30)
    axs1[0].set_xlabel('Time [hours]', fontsize=30)
    axs1[0].set_ylabel('magnitude', fontsize=20)

    # 5b. Periodogramma con linea rossa sul picco e legenda con periodo ± errore
    label = ("P = " + format_period(max_period, sigma) + " hours = "
             + format_period(max_period_day, sigma_day) + " days")
    axs1[1].plot(period1, power1, label=label)
    axs1[1].axvline(x=max_period, color='red')
    axs1[1].set_title('Periodogramma ' + str(i + 1), fontsize=30)
    axs1[1].set_xlabel('Period [hours]', fontsize=30)
    axs1[1].set_ylabel('Power', fontsize=20)
    axs1[1].legend(fontsize=20, loc=0)          # loc=0: posizione migliore

    # 5c. Curva ripiegata sul periodo trovato
    phase, profile, proferr = fold_light_curve(time, mag, errmag, max_period, nbins=nbins)

    # Il profilo è ripetuto per due cicli (fase 0-2) per rendere evidente la periodicità
    phase2 = np.concatenate([phase, phase + 1])
    profile2 = np.concatenate([profile, profile])
    proferr2 = np.concatenate([proferr, proferr])

    axs1[2].errorbar(phase2, profile2, yerr=proferr2, ls='None', marker='o', markersize=3)
    axs1[2].invert_yaxis()
    axs1[2].set_title('Folded light curve ' + str(i + 1), fontsize=30)
    axs1[2].set_xlabel('Phase', fontsize=30)
    axs1[2].set_ylabel('magnitude', fontsize=20)

    # Dimensione dei numeri sugli assi
    for ax in axs1:
        ax.tick_params(axis='x', which='major', labelsize=16)
        ax.tick_params(axis='y', which='major', labelsize=13)

    fig1.tight_layout()

# ----------------------------------------------------------------------------
# VISUALIZZAZIONE
# ----------------------------------------------------------------------------
fig.tight_layout()                 # una sola volta, dopo aver riempito tutti i pannelli
plt.show()


