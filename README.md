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


### Tips when trying to reuse the code
1. Download the Anaconda Navigator, which is a graphical user interface (GUI) included in the Anaconda distribution. It is used to easily create and manage Pythin environments, which are isolated spaces that contain specific versions of Python and its packages. Users can also search for and install new packages and libraries form the Anaconda repository, which includes a wide range of popular data science packages such as NumPy, Pandas, Scikit-learn, TensorFlow, and PyTorch. In our case we are using the following packages: ...

