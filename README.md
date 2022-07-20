# CRAFT: Character-Region Awareness For Text detection with Streamlit



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
