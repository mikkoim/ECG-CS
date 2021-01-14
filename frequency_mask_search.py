"""
Assumes you already have a model trained and datasets loaded
"""

"""# Frequency mask area search"""

from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import numpy as np

def make_mask(N,r):
    M0 = np.random.random_sample(N) < 0.5
    M0 = (M0*2)-1
    
    MM = np.zeros(N)
    
    MM[r[0]:r[1]] = 1
    M = M0*MM
    M[M == 0] = 1
    return M
    
def freq_mask(s, r):
    M = make_mask(len(s),r)
    
    x = dct(s, norm='ortho')
    x_h = x*M

    s_hat = idct(x_h, norm='ortho')
    return s_hat

s = X_test[4,0,:]
shat = freq_mask(s, (20,90))
plt.plot(s)
plt.plot(shat)

r = (20,90)
X_test_hat = np.asarray([freq_mask(x, r) for x in X_test[:,0,:]])
X_test_hat = np.expand_dims(X_test_hat,1)
plt.plot(X_test[4,0,:])
plt.plot(X_test_hat[4,0,:])

!pip install py-ecg-detectors

from ecgdetectors import Detectors
detectors = Detectors(360)

print(np.max(X_test), np.min(X_test))
s = X_test[28,0,:]
r_peaks = detectors.engzee_detector(s)

plt.plot(s)
plt.plot(r_peaks, s[r_peaks],'ro')


shat = X_test_hat[28,0,:]
r_peaks_hat = detectors.engzee_detector(shat)
plt.figure()
plt.plot(shat)
plt.plot(r_peaks_hat, shat[r_peaks_hat],'ro')

from scipy import stats
def match_peaks(R, Rhat):
  if (not R) or (not Rhat):
    return 0, 0, 0, None

  D = np.empty((len(R), len(Rhat)))

  for i, r in enumerate(R):
      for ii, ra in enumerate(Rhat):

          D[i,ii] = np.abs(r-ra);
        
  mins_Rhat = np.min(D,axis=0);
  mins_R = np.min(D,axis=1);

  precision = sum(mins_Rhat < 10)/len(Rhat);
  recall = sum(mins_R < 10)/len(R);
  f1 = stats.hmean([precision, recall])

  return precision, recall, f1, D

precision, recall, f1, D = match_peaks(r_peaks, r_peaks_hat)
print(D, precision, recall, f1)

def compare_peaks(s, shat):
  r = detectors.engzee_detector(s)
  rhat = detectors.engzee_detector(shat)
  _, _, f1, _ = match_peaks(r, rhat)
  return f1

peak_f1s = [compare_peaks(s[0].T, s[1].T) for s in zip(X_test, X_test_hat)]
print(peak_f1s)
print(np.mean(peak_f1s))

from sklearn.metrics import accuracy_score, f1_score
def mask_and_test(X,y,r):
  Xhat = np.asarray([freq_mask(x, r) for x in X[:,0,:]])
  Xhat = np.expand_dims(Xhat,1)

  # Classification
  y_pred = evaluate(Xhat, y)
  acc = accuracy_score(y, y_pred)
  f1 = f1_score(y,y_pred, average='macro')

  # Peak comparison
  peak_f1s = np.asarray([compare_peaks(s[0].ravel(), s[1].ravel()) for s in zip(X,Xhat)])
  p = np.mean(peak_f1s)
  return acc, f1, p

sp = np.arange(0,100,5)
widths = np.arange(10,100,10)

A = np.zeros((len(sp), len(widths)))
F = np.zeros_like(A)
P = np.zeros_like(A)

print(len(sp)*len(widths))

for i, s in enumerate(tqdm(sp)):
  for ii, w in enumerate(widths):
    try:
      acc, f1, p = mask_and_test(X_test, Y_test, (s,s+w))
    except:
      print('Something went wrong')
      pass
    A[i,ii] = acc
    F[i,ii] = f1
    P[i,ii] = p

A1 = A-np.min(A)
A1 = A1/np.max(A1)

P1 = P-np.min(P)
P1 = P1/np.max(P1)
print(A1)
print(P1)
OP = ((1-P1)+(A1))/2
print(OP)

plt.matshow(OP)
plt.xlabel('width')
_ = plt.xticks(range(len(widths)),widths)
plt.ylabel('Starting point')
_ = plt.yticks(range(len(sp)), sp)

minloc = np.where(OP == np.min(OP))
print(minloc)

starting_point = sp[minloc[0]]
width = widths[minloc[1]]

print(starting_point, width)
print(A[minloc])
print(P[minloc])
