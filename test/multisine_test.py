from multisine.multisine import Multisine
import numpy as np

harmonics = np.loadtxt('data/harmonics_8dec_quasi-log-8pts_no_intermod_second.txt')
base_frequency = 1
max_frequency = 100000
frequencies = harmonics * base_frequency

# Remove higher frequencies
frequencies = frequencies[0:np.where(frequencies>max_frequency)[0][0]]
frequencies[-1] = max_frequency

amplitudes = np.ones(frequencies.size) * 0.005 # V
sampling_frequency = 1000000
ms1 = Multisine(sampling_frequency, frequencies, amplitudes)
ms1.plot('voltage')
ms1.fourier_analysis(6)
ms1.plot_dft((0.01,5000))
# ms1.plot_phase((0.1,250))
# ms1.best_random_phases(30)
# ms1.save('multisine_20kHz-20mHz_8ptd_fs200kHz_sodium_optim_ampl_zero_phases.txt')