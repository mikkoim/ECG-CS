## MULTI-LEVEL REVERSIBLE ENCRYPTION FOR ECG SIGNALS USING COMPRESSIVE SENSING

This is the repository for our paper "Multi-level reversible encryption for ECG signals using compressive sensing", published in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2021.

Paper:
https://arxiv.org/pdf/2104.05325.pdf

### Abstract: 
Privacy concerns in healthcare have gained interest recently via GDPR, with a rising need for privacy-preserving data collection methods that keep personal information hidden in otherwise usable data. Sometimes data needs to be encrypted for several authentication levels, where a semi-authorized user gains access to data stripped of personal or sensitive information, while a fullyauthorized user can recover the full signal. In this paper, we propose a compressive sensing based multi-level encryption to ECG signals to mask possible heartbeat anomalies from semi-authorized users, while preserving the beat structure for heart rate monitoring. Masking is performed both in time and frequency domains. Masking effectiveness is validated using 1D convolutional neural networks for heartbeat anomaly classification, while masked signal usefulness is validated comparing heartbeat detection accuracy between masked and recovered signals. The proposed multi-level encryption method can decrease classification accuracy of heartbeat anomalies by up to 50%, while maintaining a fairly high R-peak detection accuracy.

## Licence:

Original code in this repository is licenced under the MIT License. This repository contains code from the [L1-magic](https://candes.su.domains/software/l1magic) MATLAB-package by Candes et al. and modified code from https://github.com/hsd1503/ for the 1D ResNet models.

## Citing:

If you use our code in your work, please cite:

```
@inproceedings{impio2021multi,
  title={Multi-level reversible encryption for ECG signals using compressive sensing},
  author={Impi{\"o}, Mikko and Yama{\c{c}}, Mehmet and Raitoharju, Jenni},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1005--1009},
  year={2021},
  organization={IEEE}
}
```
