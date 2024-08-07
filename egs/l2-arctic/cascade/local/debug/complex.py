import torch

# draw a sin signal
import matplotlib.pyplot as plt

x = torch.arange(0, 2 * 3.1416, 0.01)
y = torch.sin(x)
plt.plot(x, y)

# torch stft  y
y = y.unsqueeze(0)
print(y.shape)
import pdb; pdb.set_trace()
# torch.stft # (B, T) -> (B, F, T, 2)

y_stft = torch.stft(y, n_fft=1024, win_length=600, hop_length=120, center=True, normalized=False, onesided=True, return_complex=False)
y_stft_complex = torch.stft(y, n_fft=1024, win_length=600, hop_length=120, center=True, normalized=False, onesided=True, return_complex=True)

print(y_stft.shape)

