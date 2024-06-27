<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]](https://github.com/rhysdg/whisper-onnx-python/contributors)
[![Apache][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
  <h3 align="center"> Whisper ONNX: An Optimized Speech-to-Text Python Package</h2>
  <p align="center">
     <br />
    <a href="https://github.com/rhysdg/whisper-onnx-python/wiki"<strong>Explore the docs »</strong></a>
    <br />
    <br />
    <img src="data/whisper-onnx.png" align="middle" width=200>
    <br />
    <br />
    <a href="https://github.com/rhysdg/whisper-onnx-python/issues">Report Bug</a>
    .
    <a href="https://github.com/rhysdg/whisper-onnx-python/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
  * [The Story so Far](#the-story-so-far)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Scripts and Tools](#scripts-and-tools)
  * [Supplementary Data](#supplementary-data)
* [Proposed Updates](#proposed-updates)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project

### Built With

* [Onnxruntime](https://onnxruntime.ai/)


### The Story So Far

**Coming soon**



<!-- GETTING STARTED -->
## Getting Started:

- Right now getting started is as simple as either a pip install from root or the upstream repo:


  ```bash
  pip install .

  #or 

  pip install git+https://github.com/rhysdg/whisper-onnx-python.git

  ```

- For Jetpack 5 support with Python 3.11 go ahead and run the installation script first to grab a pre-built `onnxruntime-gpu` wheel for `aarch_64` and a few extra dependencies:

  ```bash
  sh jetson_install.sh 

  pip install .

  ```



## Example usage:

- Currently usage closely follows the official package but with a trt swicth (currently being debugged, False is recommended as a result) and expects either an audio file or a numy array:



  ```python
  import numpy as np
  import whisper

  args = {"language": 'English',
          "name": "small.en",
          "precision": "fp32",
          "disable_cupy": False}

  temperature = tuple(np.arange(0, 1.0 + 1e-6, 0.2))

  model = whisper.load_model(trt=False, **args)
  result = model.transcribe(
                      'data/test.wav', 
                      temperature=temperature,
                      **args
                      )
    ```

- You can also find an example voice transcription assistant at `examples/example_assistant.py`

  - Go ahead and hold in your space bar from the command line in order to start recording
  - Release to  start transcription
  - This has been tested on Ubuntu 22.04 and Jetpack 5 on a AGX  Xavier but feel free to open an issue so we can work through any issues!


  ```bash
  python examples/example_assistant.py
  ```


## Customisation:

- **Coming soon**


### Notebooks
 
- **Coming soon**

### Tools and Scripts
-  **Coming soon**


### Testing

- Ubuntu 22.04 - RTX 3080, 8-core, Python 3.11 - **passing**
- AGX Xavier, Jetpack 5.1.3, Python 3.11 - **Passing**

 - CI/CD will be expanded as we go - all general instantiation test pass so far.

### Models & Latency benchmarks


- **Coming soon**


### Similar projects

- Inspired by the work over at:
  - [whisper-onnx-tensorrt](https://github.com/PINTO0309/whisper-onnx-tensorrt)
  - [The original implementation](https://github.com/openai/whisper)

<!-- PROPOSED UPDATES -->
## Latest Updates
- Finished the core Python package

<!-- PROPOSED UPDATES -->
## Future updates

- CI/CD
- Pypi release
- Becnhmarks for Jetson devices

<!-- Contact -->
## Contact
- Project link: https://github.com/rhysdg/whisper-onnx-python
- Email: [Rhys](rhysdgwilliams@gmail.com)


<!-- MARKDOWN LINKS & IMAGES -->
[build-shield]: https://img.shields.io/badge/build-passing-brightgreen.svg?style=flat-square
[contributors-shield]: https://img.shields.io/badge/contributors-2-orange
[license-shield]: https://img.shields.io/badge/License-GNU%20GPL-blue
[license-url]: LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/rhys-williams-b19472160/
