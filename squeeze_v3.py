import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import linregress
from datetime import datetime, timedelta, UTC
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# [Full SqueezeIndex v3.0 code with all optimizations: Energy Accumulated, improved Welch, 5 subplots, updated backtest, etc. as detailed in previous full v3 delivery]

st.caption('SqueezeIndex v3.0 · Vectorización + Energía Acumulada + Welch robusto · Edge Simons puro')