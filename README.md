[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/streamlit/example-app-bug-report/main)

# CRAFT: Character-Region Awareness For Text detection with Streamlit <img alt="teaser1" src="https://camo.githubusercontent.com/442a5a932b06ec87ed75a1e355d0ed3f6a76a727ccfe096f136e47dc5b2f8b49/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f2d53747265616d6c69742d4646344234423f6c6f676f3d53747265616d6c6974266c6f676f436f6c6f723d7768697465">

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

### Author of CRAFT: Character-Region Awareness For Text detection
Fatih C. Akyon - https://github.com/fcakyon
