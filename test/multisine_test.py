from multisine.multisine import Multisine
import numpy as np

harmonics = np.loadtxt('data/harmonics_8dec_quasi-log-8pts_no_intermod_second.txt')
base_frequency = 0.1
max_frequency = 1000
frequencies = harmonics * base_frequency

# Remove higher frequencies
frequencies = frequencies[0:np.where(frequencies>max_frequency)[0][0]]
frequencies[-1] = max_frequency

amplitudes = np.ones(frequencies.size) # V
sampling_frequency = 10000
ms1 = Multisine(sampling_frequency, frequencies, amplitudes)
ms1.plot('voltage')
ms1.fourier_analysis(6)
ms1.plot_dft((0.01,500000))
# ms1.plot_phase((0.1,250))
ms1.best_random_phases(100)
ms1.normalize_waveform()
ms1.save('E:/multisine_collection/2409131232multisine_1kHz-100mHz_8ptd_fgen10kHz_random_phases_flat_normalized')
