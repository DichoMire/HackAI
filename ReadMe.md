# Cirium's HackAI Challenge #
## Southampton 2022 ##

Team: Benjamin Sanati, Emil Stoev, Miroslav Milanov

The aim of this hackaton was to develop a machine learning model, which processes data about organized events, airports and online flight query volumes and predict anomalies in the data to find which events lead to a spike in searches of particular flights.

***We have been asked to not publish the data and we respect Cirium's request. All references to the dataset have been expunged.***

We use an AutoEncoder neural network to perform automatic dimensionality reduction on the input vector and then decode it to the original size. We then compare the values of the two vectors against an activation function to detect anomalies.

![img2](https://user-images.githubusercontent.com/56360395/156949041-45a7d69d-761b-4376-b650-ce3211a712ef.png)

You can download our presentation in the files or have a look here:
https://docs.google.com/presentation/d/1I-U3-zCT2FVroj4rzkMSEnIKHSEPsJTvHFEt0GWElgE/edit?usp=sharing

Finally, here is our team at the hackaton:

![img1edited](https://user-images.githubusercontent.com/56360395/156949945-f965123d-a935-49fe-be01-94db25f680fe.png)

From left to right: Benjamin Sanati, Miroslav Milanov and Emil Stoev
