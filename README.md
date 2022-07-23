[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/streamlit/example-app-bug-report/main)

# 🔥[CRAFT: Character-Region Awareness For Text detection with Streamlit](https://ahmetenesyalcinkaya-text-detector-streamlit-main-bx1qcu.streamlitapp.com/)

## Overview

PyTorch implementation for CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.

<img alt="teaser1" src="https://github.com/AhmetEnesYalcinkaya/Text_Detector_Streamlit/blob/main/predicted.png">

## Getting started

### Installation

- Install using pip :

```console
pip install streamlit
pip install craft-text-detector
```

### Usage

```console
streamlit run main.py
```

[CRAFT libraries: Character-Region Awareness For Text detection](https://pypi.org/project/craft-text-detector/0.4.3/)

