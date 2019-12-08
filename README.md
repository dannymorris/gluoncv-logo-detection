# Object detection with GluonCV to detect logo

Credit [Natalie Rauschmayr](https://www.linkedin.com/in/rauschmayr/), a Machine Learning Scientist at AWS, for helping me get started.

This repo shows how to use GluonCV to train an object detector. GluonCV is a computer vision toolkit for rapid development and deployment of CV models. This particular model applies transfer learning to detect custom class (i.e. logo). To create a custom neural network, I use the pre-trained weights from a fast single shot detection (SSD) network with MobileNet1.0 backbone.