import streamlit as st
from streamlit import session_state as state
import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import math

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


def calculate_fft(dataset, start, end):
    fouriert = np.fft.fft(dataset) / len(dataset)  # Normalize amplitude
    tpcount = len(dataset)
    timeperiod = end - start
    values = np.arange(tpcount / 2)
    frequencies = values / timeperiod
    return fouriert, frequencies

###############################################################################
# main
###############################################################################


wavfile = st.sidebar.file_uploader(
    label='input your file', accept_multiple_files=False, key='mp3file', type=['mp3', 'wav'])
if not wavfile:
    state.frequency_list = st.sidebar.text_input(
        label='Which frequencies (in Hz and space-separated) would you like to give?',
        value='300 400 500')
    state.amplitudes_list = st.sidebar.text_input(
        label='Which amplitudes (space-separated, as many as frequencies!) would you like to give?',
        value='3 5 3')

# create sidebar widgets
st.sidebar.title("Advanced settings")
# Data options
st.sidebar.markdown("Data Options")

[tmin, tmax] = st.sidebar.slider('Select the time range', 0., 50., (0., 1.))

rate = st.sidebar.number_input(
    label='Sample points per second.', min_value=50, max_value=20000, value=1000,
    help=' You will need twice as many sampling points per second as the frequency you want to detect.')
n = (tmax - tmin) * rate

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
dt = 1 / rate
if wavfile:
    rate, data = wav.read(wavfile)
    tspan = np.arange(tmin, tmax, 1 / rate)
    nstart = math.floor(tmin * rate)
    nend = math.ceil(tmax * rate)
    at = data[nstart:nend]
else:
    tspan = np.arange(tmin, tmax, dt)
    flist = [float(x) for x in state.frequency_list.split(' ')]
    alist = [float(x) for x in state.amplitudes_list.split(' ')]
    at = np.zeros(len(tspan))
    for i in np.arange(len(flist)):
        at += alist[i] * np.sin(2 * np.pi * flist[i] * tspan)

# plot the time domain
handles["timescale"] = ax1.plot(tspan, at,
                                color='b',
                                linewidth=0.4,
                                label='timescale')[0]
ax1.set_title('Time Domain')
ax1.set_xlabel('Time (s)', horizontalalignment='right', x=1)
ax1.set_ylabel('Amplitude', horizontalalignment='right', x=0, y=1)
ax1.set_xlim([tmin, tmax])

# plot f and append the plot handle
fourierTransform, freq = calculate_fft(at, tmin, tmax)
fourierTransform_plot = fourierTransform[range(int(np.ceil(len(at) / 2)))]  # Exclude sampling frequency
handles["frequencyscale"] = ax2.plot(freq, abs(fourierTransform_plot.real), freq, abs(fourierTransform_plot.imag))
ax2.set_title('Frequency Domain')
ax2.set_xlabel('Frequency (Hz)', horizontalalignment='right', x=1)
# ax2.set_ylabel('Amplitude', horizontalalignment='right', x=1, y=1)
ax2.yaxis.tick_right()
ax2.set_xlim([0, max(freq)])
# add legend to frequencies
ax2.legend(["Real Frequency", "Imaginary Frequency"], loc='upper right')
# make all changes visible
fig.canvas.draw()

if not wavfile:
    at = at / max(at)
st.write('Clear tune')
st.audio(at, sample_rate=rate)
st.pyplot(fig)

if noise:
    at_noise = np.random.normal(at, sigma, len(tspan))
    fourierTransform_noise, freq = calculate_fft(at_noise, tmin, tmax)
    fourierTransform_noise_plot = fourierTransform_noise[range(int(np.ceil(len(at_noise) / 2)))]  # Exclude sampling frequency
    st.write('Disturbed tune with a random noise of ' + str(sigma))
    handles["timescale"].set_ydata(at_noise)
    handles["frequencyscale"][0].set_ydata(abs(fourierTransform_noise_plot.real))
    handles["frequencyscale"][1].set_ydata(abs(fourierTransform_noise_plot.imag))
    fig.canvas.draw()
    st.audio(at_noise, sample_rate=rate)
    st.pyplot(fig)

if noise or wavfile:
    cap = st.sidebar.number_input('Cap for filter', min_value=0.01, max_value=0.99)
    if noise:
        fourierTransform_filtered = fourierTransform_noise
    else:
        fourierTransform_filtered = fourierTransform
    fourierTransform_filtered[fourierTransform_filtered < cap*max(fourierTransform_filtered)] = 0
    st.write('Filtered tune capping off all frequencies with an amplitude below ' + str(100 * cap) + str(fourierTransform_filtered[abs(fourierTransform_filtered) < cap]))
    at_filtered = np.fft.ifft(fourierTransform_filtered)
    fourierTransform_filtered_plot = fourierTransform_filtered[range(int(np.ceil(len(at) / 2)))]  # Exclude sampling frequency
    handles["timescale"].set_ydata(at_filtered.real)
    handles["frequencyscale"][0].set_ydata(abs(fourierTransform_filtered_plot.real))
    handles["frequencyscale"][1].set_ydata(abs(fourierTransform_filtered_plot.imag))
    fig.canvas.draw()
    st.audio(at_filtered, sample_rate=rate)
    st.pyplot(fig)
