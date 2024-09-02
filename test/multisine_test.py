from multisine.multisine import Multisine
import numpy as np

harmonics = np.loadtxt('data/harmonics_8dec_quasi-log-8pts_no_intermod_second.txt')
base_frequency = 0.01
max_frequency = 10000
frequencies = harmonics * base_frequency

# Remove higher frequencies
frequencies = frequencies[0:np.where(frequencies>max_frequency)[0][0]]
frequencies[-1] = max_frequency

amplitudes = np.ones(frequencies.size) * 0.025 # V
sampling_frequency = 100000
ms1 = Multisine(sampling_frequency, frequencies, amplitudes)
ms1.plot('voltage')
ms1.fourier_analysis(6)
ms1.plot_dft((0.01,500000))
# ms1.plot_phase((0.1,250))
ms1.normalize_waveform()
ms1.best_random_phases(10)
ms1.save('E:/multisine_collection/2408291723multisine_10kHz-10mHz_8ptd_fgen100kHz_random_phases_amplitude_from_lithium_exp_normalized')
