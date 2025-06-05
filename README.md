<h1 align="center">PnP-Nystra : Plug-and-Play Linear Attention for Pre-Trained Image and Video Restoration Models</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---
<p align="center">
  PnP-Nystra is a Nyström-based, <strong>training-free</strong> replacement for MHSA in image/video restoration models (e.g., SwinIR, Uformer, RVRT). It delivers <strong>2–4× GPU</strong> and <strong>2–5× CPU</strong> inference speedups with under 1.5 dB PSNR loss on denoising, deblurring, and super-resolution tasks.
  <br>
</p>


## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Deployment](#deployment)
- [Usage](#usage)
- [Built Using](#built_using)
- [TODO](../TODO.md)
- [Contributing](../CONTRIBUTING.md)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
Write about 1-2 paragraphs describing the purpose of your project.

## 🏁 Getting Started <a name = "getting_started"></a>

### Prerequisites

1. **Clone the repository**

   ```bash
   git clone https://github.com/Srinivas-512/PnP_Nystra
   cd PnP_Nystra
   ```

2. **Download pretrained models**

   * Go to the Google Drive link for pretrained models:

     ```
     https://drive.google.com/drive/folders/1G0jb_yN6aYotbcPNhWj_3JdAOjzgLFb8?usp=sharing
     ```
   * Download the weight folders for the models and extract their contents.
   * Place everything under:

     ```
     ./pretrained_models/
     ```

     After this step you should have the following structure:

     ```
     PnP_Nystra/
       └─ pretrined_models/
           ├─ RVRT/
           └─ SwinIR/
           └─ Uformer/
     ```

3. **Download datasets**

   * Go to the Google Drive link for datasets:

     ```
     https://drive.google.com/drive/folders/1abRCvUDrrRYnxjzhzkdbZIgJAKabGv7T?usp=sharing
     ```
   * Download the dataset folders for the models and extract their contents.
   * Place the dataset files under:

     ```
     ./datasets/
     ```

     After this step you should have the following structure:

     ```
     PnP_Nystra/
       └─ datasets/
           ├─ RVRT/
           ├─ SwinIR/
           └─ Uformer/
     ```


### Installing
A step by step series of examples that tell you how to get a development env running.

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## 🔧 Running the tests <a name = "tests"></a>
Explain how to run the automated tests for this system.

### Break down into end to end tests
Explain what these tests test and why

```
Give an example
```

### And coding style tests
Explain what these tests test and why

```
Give an example
```

## 🎈 Usage <a name="usage"></a>
Add notes about how to use the system.

## 🚀 Deployment <a name = "deployment"></a>
Add additional notes about how to deploy this on a live system.

## ⛏️ Built Using <a name = "built_using"></a>
- [MongoDB](https://www.mongodb.com/) - Database
- [Express](https://expressjs.com/) - Server Framework
- [VueJs](https://vuejs.org/) - Web Framework
- [NodeJs](https://nodejs.org/en/) - Server Environment

## ✍️ Authors <a name = "authors"></a>
- [@kylelobo](https://github.com/kylelobo) - Idea & Initial work

See also the list of [contributors](https://github.com/kylelobo/The-Documentation-Compendium/contributors) who participated in this project.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- Hat tip to anyone whose code was used
- Inspiration
- References
