# Data-Augmentation-in-computer-vision

### Introduction
As you’ve seen, a well-designed data augmentation setup is crucial for the efficient use of training data in deep learning. For image-based problems, several standard methods augment the training set, e.g., rotations, flips, translation, and more. However, there’s a continuous stream of research and development aimed at creating more advanced, and ideally more powerful, data augmentation techniques. This project will explore these developments and provide a careful setup for evaluating the many methods: their strengths and weaknesses. It will also attempt to clarify what problems and situations fit with what data augmentation techniques and assist other developers in their choice and design of data augmentation setups. 

### Goals
- Explore the many existing techniques for data augmentation in deep learning for computer 
vision 
- Design an experimental setup where you can evaluate the effect of applying these methods on 
a fixed set of data sets and problems. 
- Create a library, or at least some very well-thought-out examples, that make it easy for others to reuse what you’ve done in this project on their own image data.  

### Methods and materials
- Data: The image data sets are listed under “External datasets” in fastai. For example, the 
IMAGENETTE data set. 
- Methods: through the fastai course and library, you have been exposed to some basic data 
augmentation techniques (rotation, flips, image intensity variation) and some more advanced 
methods (MixUp, CutMix, and more). A substantial part of this project will be to explore, try, 
and perhaps extend what’s out there for data augmentation in computer vision. 

### What’s expected 
At the project deadline, you'll hand in source code, documentation, and potentially other 
artifacts from your work. Precisely what is partly up to you: 
- Required: A link to a well-documented online Git repository containing all the code and 
documentation necessary to understand and reproduce your work, including code to 
create and run any applications you've made. You also must explain the larger context 
to which your work belongs, for example, in a readme file in your repository. 
- Recommended: An application running in the cloud or that can be installed on a 
computer or a mobile device. The setup from Module 2 will get you quite far (Gradio + 
HuggingFace Spaces or similar), but feel free to be more ambitious.  
- Recommended: A blog post explaining the background of your project, its objectives, 
what methods and data you used, and what you achieved. The post should target a 
broad audience. For example, other software engineering students with little knowledge 
of the application domain and the relevant methods. fastpages is a good blogging 
solution.  

### Folder structure:

    C:Data-Augmenation-in-computer-vision
    │   README.md
    │   req.txt
    │
    ├───data
    │       BH_n4_M10_res50_15000_events.h5
    │       PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL_res50_15000_events.h5
    │       
    ├───flagged
    │   │   log.csv
    │   │   
    │   ├───image
    │   │       tmp7s6x02be.png
    │   │       tmpdv9mt9tr.png
    │   │       
    │   ├───output
    │   │       tmp1re8hz8m.json
    │   │       tmp5whinqps.json
    │   │       
    │   └───output 0
    │           tmpf761f5br.json
    │           tmpvycvh8ue.json
    │
    ├───methods
    │   │   app.py
    │   │   dataloader.py
    │   │   nnmodel.py
    │   │   plotCreator.py
    │   │   trainer.py
    │   │   trainer2.py
    │   │
    │   ├───models
    │   │       best_model.pth
    │   │       best_model2.pth
    │   │
    │   └───__pycache__
    │           dataloader.cpython-39.pyc
    │           nnmodel.cpython-39.pyc
    │           trainer2.cpython-39.pyc
    │
    └───notebooks
        │   CrtImageFromh5toPNG.ipynb
        │   TrainAndSave.ipynb
        │   Start.ipynb
        │   Testing.ipynb
        │   TestOnDiffDataAugm.ipynb
        │
        ├───models
        │       best_model.pth
        │       best_model2.pth
        │
        └───output
            ├───black_holes
            │       bh_0.png
            │       ...
            │       bh_99.png
            │
            └───sphalerons
                    sph_0.png
                    ...
                    sph_99.png


### Folder structure explained
C:Data-Augmenation-in-computer-vision
│   README.md (Readme file of the whole project explaining what, where, and how all works in a summarized way)
│   req.txt (Contains all libraries and packages needed for Anaconda environment)
│
├───data
│       BH_n4_M10_res50_15000_events.h5 (Data of black holes in h5 histogram format)
│       PP13-Sphaleron-THR9-FRZ15-NB0-NSUBPALL_res50_15000_events.h5 (Data for sphalerons in h5 histograms format)
│       
├───flagged (Not important file, it's just something that is being created when you run code)
│   │   log.csv
│   │   
│   ├───image
│   │       tmp7s6x02be.png
│   │       tmpdv9mt9tr.png
│   │       
│   ├───output
│   │       tmp1re8hz8m.json
│   │       tmp5whinqps.json
│   │       
│   └───output 0
│           tmpf761f5br.json
│           tmpvycvh8ue.json
│
├───methods
│   │   app.py (app to start Gradio interface web application to load and submit images for classifying)
│   │   dataloader.py (contains some plot and data loading functions for notebooks)
│   │   nnmodel.py (where all our models reside)
│   │   plotCreator.py
│   │   trainer.py (training which returns plot of all trainings, perfect to train on 1 data set)
│   │   trainer2.py (better train methods which returns all metrics that can be used to combine them in one smooth plot)
│   │
│   ├───models (these are our saved models)
│   │       best_model.pth (SymmetricNet2)
│   │       best_model2.pth (SymmetricNet2)
│   │       best_model3.pth (VGGNet2)
│   │
│   └───__pycache__
│           dataloader.cpython-39.pyc
│           nnmodel.cpython-39.pyc
│           trainer2.cpython-39.pyc
│
└───notebooks
    │   CrtImageFromh5toPNG.ipynb (this notebook lets you create images from dataset, it's not the best but gives images that can be used more or less for Gradio app)
    │   TrainAndSave.ipynb (here we train with all combined data augmentation techniques)
    │   Start.ipynb (Just some start code we had in the beginning)
    │   Testing.ipynb (Testing code on png images to see how it can classify images from png format)
    │   TestOnDiffDataAugm.ipynb (Testing different data augmentation techniques with several runs and their average is then plotted to see what techniques change accuracy)
    │
    ├───models (same as previous)
    │       best_model.pth
    │       best_model2.pth
    |       best_model3.pth
    │
    └───output (This is where our images reside)
        ├───black_holes
        │       bh_0.png
        │       ...
        │       bh_99.png
        │
        └───sphalerons
                sph_0.png
                ...
                sph_99.png


 
### Tips when trying to reuse the code
1. Install necessary apps, VSCode and anaconda from:

https://docs.anaconda.com/free/anaconda/install/

https://code.visualstudio.com/download

2. Import project to your vscode enviroment
    (You can see that my tree structure is a bit different since i have folders data and flagged but they are not necessary to test gradio interface i created to classify images, but they are necessary to run training on models and evaluation, reason they are nto uploaded since they are out of allowed memory for upload on Github)

2. Import the project into your VSCode environment. Note that my folder structure is slightly different as it includes the "data" and "flagged" folders, which are not necessary to test the Gradio interface I created for classifying images. However, they are necessary for training and evaluating the models. These folders are not uploaded to GitHub due to memory constraints.

3. Once Anaconda is installed, launch it. A terminal window resembling a Bash script will pop up. Don't worry, it's just how Anaconda works. After a while, you will see a green circle and the application will start. Now, navigate to your VSCode, open the terminal in VSCode, ensure it is in the project directory, and run the following command:

$ conda create -n <environment-name> --file req.txt

Replace <environment-name> with the desired name for your environment. It may take some time to download all the required packages.

3. Now, you can open the app.py file and run it. In the terminal, you will see a web application address that looks like this: http://127.0.0.1:7860/. Ctrl-Click on it to open the interface. Next, open the "output" folder, navigate to "black_holes" or "sphalerons," and submit images to the classifier to see if they are classified correctly.

4. The results might not be as accurate as expected, not because the model is bad, but due to differences in the structure and creation of the images used for training compared to the PNG files. Unfortunately, I couldn't get the images transformed with the exact same colors as the original data and this affects how well model can classify those images However, this gives you an idea of how the model predicts and allows you to easily try it out. You can also explore different notebooks where all the training, plots (both in graph and tabular format), and data reside.

