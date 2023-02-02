import streamlit as st
from streamlit import session_state as state
import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

###############################################################################
# main
###############################################################################
file_input = st.sidebar.checkbox(label='Upload a file', value=False)
if file_input:
    wavfile = st.sidebar.file_uploader(
        label='input your file', accept_multiple_files=False, key='mp3file', type=['mp3', 'wav'])
else:
    state.frequency_list = st.sidebar.text_input(label='Which frequencies (in Hz and space-separated) would you like to give?',
                                                 value='30 40 50')
    state.amplitudes_list = st.sidebar.text_input(
        label='Which amplitudes (space-separated, as many as frequencies!) would you like to give?',
        value='1 2 3')

# create sidebar widgets
st.sidebar.title("Advanced settings")
# Data options
st.sidebar.markdown("Data Options")

tmin = st.sidebar.number_input(label='Starting time', min_value=0, max_value=50, value=0, key='tmin')

tmax = st.sidebar.number_input(label='Ending time', min_value=tmin + 1, max_value=100, value=1, key='tmax')

n = st.sidebar.number_input(
    label='Sample points. You will need twice as many sampling points per second as the frequency you want to detect.'
          'E.g., for 2 seconds record time you need 2000 sample points to detect frequencies up to 250 Hz.',
    min_value=50, max_value=20000, value=1000)

noise = st.sidebar.checkbox(label="Use random noise generator", value=False)

if noise:
    sigma = st.sidebar.number_input(label='sigma', min_value=0., max_value=10000., value=0.1, key='sigma')
else:
    sigma = 0

# Visualization Options
st.sidebar.markdown("Visualization Options")

# Good for in-classroom use
qr = st.sidebar.checkbox(label="Display QR Code", value=False)

###############################################################################
# Create main page widgets
if qr:
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.title('Demonstration of Discrete Fourier Transformation')
    with tcol2:
        st.markdown('## <img src="https://api.qrserver.com/v1/create-qr-code/?size=150x150&data='
                    'https://share.streamlit.io/PhiSpel/spielbeispiel-dft/main" width="200"/>',
                    unsafe_allow_html=True)
else:
    st.title('Demonstration of Discrete Fourier Transformation')

#######################
# Initialize the plot #
#######################

handles = {}
fig, ax = plt.subplots(1, 2)
ax1 = fig.axes[0]
ax2 = fig.axes[1]

# calculate data
dt = (tmax - tmin) / n
if not file_input:
    tspan = np.arange(tmin, tmax, dt)
    flist = [float(x) for x in state.frequency_list.split(' ')]
    alist = [float(x) for x in state.amplitudes_list.split(' ')]
    at = np.zeros(len(tspan))
    for i in np.arange(len(flist)):
        at += alist[i] * np.sin(2 * np.pi * flist[i] * tspan)
    if noise:
        at = np.random.normal(at, sigma, len(tspan))
else:
    rate, data = wav.read(wavfile)
    tspan = np.arange(tmin, tmax, 1 / rate)
    nstart = tmin * rate
    nend = tmax * rate
    at = data[nstart:nend]
    # fourierTransform = fft(data)

# plot the time domain
handles["timescale"] = ax1.plot(tspan, at,
                                color='b',
                                linewidth=0.4,
                                label='timescale')[0]
ax1.set_title('Time Domain')
ax1.set_xlabel('Time (s)', horizontalalignment='right', x=1)
ax1.set_ylabel('Amplitude', horizontalalignment='right', x=0, y=1)
ax1.set_xlim([tmin, tmax])
# ax1.set_ylim([-a1 - a2 - a3, a1 + a2 + a3])

# scale_to_frequency
fourierTransform = np.fft.fft(at) / len(at)  # Normalize amplitude
fourierTransform = fourierTransform[range(int(np.ceil(len(at) / 2)))]  # Exclude sampling frequency
tpCount = len(at)
timePeriod = tmax - tmin
values = np.arange(tpCount / 2)
freq = values / timePeriod

# plot f and append the plot handle
handles["frequencyscale"] = ax2.plot(freq, abs(fourierTransform.real), freq, abs(fourierTransform.imag))
ax2.set_title('Frequency Domain')
ax2.set_xlabel('Frequency (Hz)', horizontalalignment='right', x=1)
# ax2.set_ylabel('Amplitude', horizontalalignment='right', x=1, y=1)
ax2.yaxis.tick_right()
ax2.set_xlim([0, max(freq) / 2])
# add legend to frequencies
ax2.legend(["Real Frequency", "Imaginary Frequency"], loc='upper right')
# make all changes visible
fig.canvas.draw()

st.pyplot(fig)
