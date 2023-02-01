import streamlit as st
# caching option only for reset-button
# from streamlit import caching

import numpy as np
# import math

import matplotlib.pyplot as plt
# from scipy import fft

# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
import matplotlib.font_manager

# make sure the humor sans font is found. This only needs to be done once
# on a system, but it is done here at start up for usage on share.streamlit.io.
matplotlib.font_manager.findfont('Humor Sans', rebuild_if_missing=True)

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


#############################################
# Define the function that updates the plot #
#############################################

# To Do: Why does caching update_plot hang?
# @st.cache(suppress_st_warning=True)
def update_plot(x1, y1):
    # Creates a Matplotlib plot if the dictionary st.session_state.handles is empty, otherwise
    # updates a Matplotlib plot by modifying the plot handles stored in st.session_state.handles.
    # The figure is stored in st.session_state.fig.

    x2, y2 = scale_to_frequency(x1, y1)

    handles = st.session_state.handles

    ax1 = st.session_state.mpl_fig.axes[0]
    ax2 = st.session_state.mpl_fig.axes[1]

    # if the dictionary of plot handles is empty, the plot does not exist yet. We create it. Otherwise the plot exists,
    # and we can update the plot handles in fs, without having to redraw everything (better performance).
    if not handles:
        #######################
        # Initialize the plot #
        #######################

        # plot the data points
        handles["timescale"] = ax1.plot(1, 1,
                                        x1, y1,
                                        color='g',
                                        linewidth=0,
                                        marker='o',
                                        ms=1,
                                        label='timescale')[0]  # .format(degree))[0]

        # plot f and append the plot handle
        handles["frequencyscale"] = ax2.plot(1, 2,
                                             x2, y2,
                                             color='b',
                                             label="frequencyscale")[0]

        ###############################
        # Beautify the plot some more #
        ###############################

        plt.title('Approximation of a series of data points')
        plt.xlabel('x', horizontalalignment='right', x=1)
        plt.ylabel('y', horizontalalignment='right', x=0, y=1)

        # set the z order of the axes spines
        # for k, spine in ax.spines.items():
        #    spine.set_zorder(0)

        # set the axes locations and style
        # ax.spines['top'].set_color('none')
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['right'].set_color('none')

    else:
        ###################
        # Update the plot #
        ###################

        # Update the data points plot
        handles["timescale"].set_xdata(x1)
        handles["timescale"].set_ydata(y1)

        # update the input plot
        handles["frequencyscale"].set_xdata(x2)
        handles["frequencyscale"].set_ydata(y2)

    # set x and y ticks, labels and limits respectively
    xticks = []
    xticklabels = [str(x) for x in xticks]
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    yticks = []
    yticklabels = [str(x) for x in yticks]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)

    # set the x and y limits
    ax1.set_xlim([min(x1) - 0.5, max(x1) + 0.5])
    ax1.set_ylim([min(y1) - 0.5, max(y1) + 0.5])

    # show legend
    legend_handles = [handles["timescale"]]
    ax1.legend(handles=legend_handles,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.15),
               ncol=2)

    # make all changes visible
    st.session_state.mpl_fig.canvas.draw()


def scale_to_frequency(x1, y1):
    x2 = np.linspace(0, n)
    y2 = np.fft.fft(y1)
    return x2, y2


def clear_figure():
    del st.session_state['mpl_fig']
    del st.session_state['handles']


###############################################################################
# main
###############################################################################
# create sidebar widgets

st.sidebar.title("Advanced settings")

# Data options
st.sidebar.markdown("Data Options")

n = st.sidebar.number_input(
    'resolution',
    min_value=500,
    max_value=5000,
    value=1000)

tmin = st.sidebar.number_input('tmin',
                               min_value=0,
                               max_value=50,
                               value=0)

tmax = st.sidebar.number_input('tmax',
                               min_value=0,
                               max_value=50,
                               value=10)

# Visualization Options
st.sidebar.markdown("Visualization Options")

# Good for in-classroom use
qr = st.sidebar.checkbox(label="Display QR Code", value=False)

# for now, I will assume matplotlib always works and we dont need the Altair backend
# backend = 'Matplotlib' #st.sidebar.selectbox(label="Backend", options=('Matplotlib', 'Altair'), index=0)

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

col1, col2, col3 = st.columns(3)
with col1:
    file_input = st.file_uploader(
        label='input your file')  # , type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")
    # st.text_input(label='input your function',
    #                        value='0.2*x**2 + 0.5 - x*math.sin(x)',
    #                        help='''type e.g. 'math.sin(x)' to generate a sine function''')

# initialize the Matplotlib figure and initialize an empty dict of plot handles
if 'mpl_fig' not in st.session_state:
    st.session_state.mpl_fig, st.session_state.axes = plt.subplots(1, 2, figsize=(8, 3))

if 'handles' not in st.session_state:
    st.session_state.handles = {}

time_scaled_x = np.linspace(0, n)
time_scaled_y = np.exp(2j * np.pi * time_scaled_x / 8)

# update plot
update_plot(time_scaled_x, time_scaled_y)
st.pyplot(st.session_state.mpl_fig)
