import streamlit as st
from streamlit import session_state as state
import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import math
import wave

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Visualization Options
st.sidebar.markdown("Visualization Options")

# Good for in-classroom use
qr = st.sidebar.checkbox(label="Display QR Code", value=False)

def calculate_fft(dataset, start, end):
    fouriert = np.fft.fft(dataset) / len(dataset)  # Normalize amplitude
    tpcount = len(dataset)
    timeperiod = end - start
    values = np.arange(tpcount / 2)
    frequencies = values / timeperiod
    return fouriert, frequencies

def read_wavfile():
    rate, data = wav.read(wavfile)
    tspan = np.arange(tmin, tmax, 1 / rate)
    nstart = math.floor(tmin * rate)
    nend = math.ceil(tmax * rate)
    at = data[nstart:nend]

###############################################################################
# Sidebar
###############################################################################

st.sidebar.subheader('Upload a file')
wavfile = st.sidebar.file_uploader('Your file', accept_multiple_files=False, key='mp3file', type=['wav'], label_visibility='collapsed')

st.sidebar.subheader('Advanced settings')
tlim = st.sidebar.number_input('Maximum time', min_value=0., max_value=100., step=0.1, value=2.)

rate = st.sidebar.number_input(
    label='Sample points per second.', min_value=100, max_value=40000, value=4000,
    help='You will need twice as many sampling points per second as the frequency you want to detect.')

###############################################################################
# Create main page widgets
###############################################################################
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

with st.expander('Input your sound parameters'):
    col1, col2, col3 = st.columns(2)
    with col1:
        state.frequency_list = st.text_input(
            label='Which frequencies (in Hz and space-separated) would you like to give?',
            value='300 400 500')
    with col2:
        state.amplitudes_list = st.text_input(
            label='Which amplitudes (space-separated, as many as frequencies!) would you like to give?',
            value='3 5 3')
    [tmin, tmax] = st.slider('Select the time range to be analyzed', 0., tlim, (0.5, 0.6))
    with col3:
        default_wavfile = st.checkbox('Show me Stars')
        if default_wavfile:
            wavfile = wave.open('StarWars60.wav', 'r')
            [tmin, tmax] = [0., 3.]

#######################
# Initialize the plot #
#######################

handles = {}
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax1 = fig.axes[0]
ax2 = fig.axes[1]

# calculate data
dt = 1 / rate
n = (tmax - tmin) * rate
if wavfile:
    read_wavfile()
else:
    tspan = np.arange(tmin, tmax, dt)
    flist = [float(x) for x in state.frequency_list.split(' ')]
    alist = [float(x) for x in state.amplitudes_list.split(' ')]
    at = np.zeros(len(tspan))
    for i in np.arange(len(flist)):
        at += alist[i] * np.sin(2 * np.pi * flist[i] * tspan)

# plot the time domain
handles["timescale"] = ax1.plot(tspan, at)
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

#if not wavfile:
    #at = at / max(at)
st.write('Clear tune')
with st.expander('Plot with true sound', expanded=True):
    col1, col2 = st.columns([1,3])
    with col2:
        st.audio(at, sample_rate=rate)
    st.pyplot(fig)

with st.expander('Added random noise'):
    col1, col2 = st.columns([1,3])
    with col1:
        sigma = st.number_input(label='sigma', min_value=0., max_value=10000., value=4., step=0.1, key='sigma', label_visibility='collapsed', help='Select sigma of the randomized noise')
    at_noise = np.random.normal(at, sigma, len(tspan))
    fourierTransform_noise, freq = calculate_fft(at_noise, tmin, tmax)
    fourierTransform_noise_plot = fourierTransform_noise[range(len(freq))]  # Exclude sampling frequency
    st.write('Disturbed tune with a random noise of ' + str(sigma))
    handles["timescale"][0].set_ydata(at_noise)
    handles["frequencyscale"][0].set_ydata(abs(fourierTransform_noise_plot.real))
    handles["frequencyscale"][1].set_ydata(abs(fourierTransform_noise_plot.imag))
    fig.canvas.draw()
    with col2:
        st.audio(at_noise, sample_rate=rate)
    st.pyplot(fig)

# if noise or wavfile:
with st.expander('Filtered plot'):
    col1, col2 = st.columns([1,3])
    with col1:
        cap = st.number_input('Cap for filter', min_value=0.01, max_value=0.99, value=0.2, label_visibility='collapsed', help='Select cap-off for the filter')
    fourierTransform_filtered = fourierTransform_noise
    if wavfile:
        fourierTransform_filtered = fourierTransform
    # Amplitude-wise cap-opp
    abscap = cap*max(np.absolute(fourierTransform_filtered))
    fourierTransform_filtered[np.absolute(fourierTransform_filtered) < abscap] = 0
    # Frequency cap-off
    # fourierTransform_filtered[freq > 1000] = 0
    fourierTransform_filtered_plot = fourierTransform_filtered[range(len(freq))]  # Exclude sampling frequency
    at_filtered = np.fft.ifft(fourierTransform_filtered)#, n=len(tspan))
    at_filtered = at_filtered / max(at_filtered) * max(at_noise)    
    # update plots
    handles["timescale"][0].set_ydata(at_filtered.real)
    handles["timescale"].append(ax1.plot(tspan, at_filtered.imag))
    ax1.legend(["Real Frequency", "Imaginary Frequency"], loc='upper right')
    handles["frequencyscale"][0].set_ydata(abs(fourierTransform_filtered_plot.real))
    handles["frequencyscale"][1].set_ydata(abs(fourierTransform_filtered_plot.imag))
    st.write('Filtered tune capping off all frequencies with an amplitude below an amplitude of ' + str(round(abscap,2)))
    fig.canvas.draw()
    with col2:
        st.audio(at_filtered, sample_rate=rate)
    st.pyplot(fig)
