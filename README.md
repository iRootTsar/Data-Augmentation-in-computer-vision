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
