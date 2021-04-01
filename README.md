<font color  = "red">
    
### Important read
For best results, please see this iPythonNotebook on jupyterlab or on jupyter-notebook in **light mode** for best results.
This is because  the rendered equations will be hard to see on a dark blackground and also there might be some alignment issues.
</font>

# **Few shot adversarial learning of Realistic Neural Talking Head Models**
## Explaination, Implementations and Discussion

When [MyHeritage.com](https://www.myheritage.com) announced the release of [Deep Nostalgia](https://www.myheritage.com/deep-nostalgia), a tool to animate faces from still photos, it received an overwhelming response from people. The website reported that over 1 million photos were animated in the first 48 hours alone and was reaching 3 million on the third day ([source](https://blog.myheritage.com/2021/02/deep-nostalgia-goes-viral/)). The world was going gaga over this and honestly, rightly so. It is only natural for a human to grab any chance they have at seeing their lost loved ones living again and when I found out about this tool while scrolling through my [reddit](https://www.reddit.com) feed, I also did not hesitate to jump on the bandwagon. Here is an example of a photo animated using Deep Nostalgia:

![example of Deep Nostalgia](https://media.giphy.com/media/vNEulvl9tMP2GDKpoz/giphy.gif) ![Another example](https://media.giphy.com/media/JRnVuIZKw5tPFPgdaH/giphy.gif)

I tried this tool on a photo of my grandfather who passed in the year 2003 and there he was, moving, looking around, smiling almost naturally. I showed my mother the animation on her birthday a few days later and I could see first-hand what the people meant in the reviews they gave this tool. Being a student of Artificial Intelligence, I couldn't help but wonder how this tool worked so I started looking for the tech behind it. It led me to the company that developed it: [D-ID](https://www.d-id.com/). D-ID is a company which started with anonymization of facial featuress in photos using AI and went on to develop [Talking Heads and the Live Portrait](https://www.d-id.com/reenactment/) which are a part of its reenactment suite.

Now, obviously the tech is proprietary so I could not find much details but I came across a [similar paper](https://arxiv.org/abs/1905.08233) from researchers at the Samsung AI center, Moscow. This blog post is going to be about this paper from the samsung AI lab: _Few-Shot Adversarial Learning of Realistic Neural Talking Head Models_.

Feeling overwhelmed by the title of the paper? I was too. Let's change that.

# Overview

The researchers at the Samsing AI lab developed a [GAN (Generativce Adversarial Network)](https://arxiv.org/abs/1406.2661) to make realistic talking heads from photos of people with as few learning shots as possible. The model works by taking a source image and a target video/image and extracts the landmarks from the target image and applyting them to the source image, animating the source image into a talking head image. So, when applied, the implementation looks something like: 

Driver video / Training video: <center> ![driver image](https://media.giphy.com/media/5AXoAsfkw7ESOYzPH6/giphy.gif) <center> 
    
Extracted landmarks:<center>![landmarks gif](https://media.giphy.com/media/4V6KKfZT0OlOCWEN7z/giphy.gif)</p>
Now the extracted landmarks were applied to this set of images:<br><center>![trainign shots](https://i.ibb.co/hM3Hq8y/IMG-5213.jpg)</center> so the resulting animated talking head came out to be: </center><br><center>![result](https://media.giphy.com/media/6siZvkfigPHnX9zQdA/giphy.gif)</center>

Compare the source, the landmarks and the result:<center>
![driver image](https://media.giphy.com/media/5AXoAsfkw7ESOYzPH6/giphy.gif) ![landmarks gif](https://media.giphy.com/media/4V6KKfZT0OlOCWEN7z/giphy.gif) ![result](https://media.giphy.com/media/6siZvkfigPHnX9zQdA/giphy.gif)</center>


# BUT HOWWWW??
### Underlying concepts
#### 1. GAN - Generative Adversarial Network


The underlying principle of such systems in the recent times has been GAN's, or Generative Adversarial Networks. GAN's are, in simple words, recent machine learning frameworks developed by [Ian J. Goodfellow et. al.](https://arxiv.org/abs/1406.2661) in 2014 which can generate new data that is similar to the trainng data but also quite different different at the same time. Nvidia Corp. recently made a GAN which generated human faces that do not exist in real life. The implementation can be seen on www.thispersondoesnotexist.com. Every time you reload this page, a new face appears which belongs to nobody. A creepy choice of words, I know, but interesting nonetheless. Surprisingly, these GAN's have somehow overcome the [Uncanny Valley Effecy](https://spectrum.ieee.org/automaton/robotics/humanoids/what-is-the-uncanny-valley) and look real.

GAN's have two parts: 
- A Generator 
- A Discriminator

An informal definition of generators would be a "model that can use the training data and use that to genenrate new data" and it has to do it as effectively as possible, and informally, a discriminator would be a "model that has to differentiate between the original data and the generated data and provide feedback to the generator".

In even simpler words, it's like a game of hide-and-seek; the generator has to generate new data and try to fool the discriminator, while the discriminator has to try its best to not be fooled by the data generated by the generator.

Mathematically,

> **Generators** have to calculate the joint probability, that is,  _P(a, b, c, ...)_ if there are multiple labels or _P(x)_ if there is only one label

whereas

> **Discriminators** have to calculate only the conditional probability, that is, _P(a | b)_.

#### 2. Embedder

In the paper, an embedder network is also used in the meta learning stage. An embedder network outputs the embedding vectors wich are then fed into the generator and help in creating new data. The equation for generating embedding vectors is:
  
<center>  
<img src = "https://render.githubusercontent.com/render/math?math=\hat{e}_{i} = \frac{1}{K}\sum_{k=1}^{K}E(x_{i}(s_{k}), y_{i}(s_{k}),\phi)"><br></center>

where 
<img src = "https://render.githubusercontent.com/render/math?math=\hat{e}_{i}"> is the embedding vector, that is being calculated by taking an average of the ouptut of the **_E(x)_** output. K is the number of episodes in K-shot learning. For few shot learning, the value of K is low. Here **_i_** is a randomly chosen video from the training set and **_t_** is a randomly chosen frame from the selected training video. The network parameters that are learned during this meta-learning stage are stored in <img src = "https://render.githubusercontent.com/render/math?math=\phi">.

In english, **an embedder network is used to extract information from training videos and the landmarks and feed the learned parameters (phi) from the training videos to the Generator network.**

# Okaayyyy.. go off, I guess?
### I have the general idea now but what about it? how is everything implemented??

#### 1. Generator
In the previous step, we got the embedding vector, which will be fed into the generator network. 

The Generator takes the landmark image of a different frame, the _embedding vector_ from the embedder network and a video frame. The output is a new frame genrated by the Generator. The generator is represented as:


<center>
<img src = "https://render.githubusercontent.com/render/math?math=\hat{x}_{i}(t) = G(y_{i}(t), \hat{e}_{i}, \Psi, P)"><br></center>

where <br>
<img src = "https://render.githubusercontent.com/render/math?math=\y_{i}(t)"> is the landmark image, <br> 
<img src = "https://render.githubusercontent.com/render/math?math=\hat{e}_{i}"> is the embedding vector, <br>
<img src = "https://render.githubusercontent.com/render/math?math=\psi"> are the _person-generic_ parameters that are learnt during meta-learning itself and <br>
<img src = "https://render.githubusercontent.com/render/math?math=\hat{\psi}_{i}"> are the _person-specific_ parameters trained from the embedding vectors. <br>
The output is <img src = "https://render.githubusercontent.com/render/math?math=\hat{x}_{i}"> which is a synthesized video frame.

Once we have the synthesized output video, it is then sent to the discriminator to predict whether it is a real image or a synthesized video.

#### 2. Discriminator

The discriminator, used to differentiate between real and fake outputs form the generator, takes an input frame from video sequence either from the generator output or from the training dataset, the index of the video from which the frame is taken. The output is _r_, a.k.a realism score which is a single scalar value indicating how much the discriminator "_thinks_" that the frame is real. Yes, i really made that pun.

representation of a discriminator is:

<center>  
<img src = "https://render.githubusercontent.com/render/math?math=r = D(x_{i}(t), y_{i}(t),i, \theta, W, w_{0}, b)"><br></center>

where,<br>
<img src = "https://render.githubusercontent.com/render/math?math=x_{i}(t)">  is the video frame, <br>
<img src = "https://render.githubusercontent.com/render/math?math=y_{i}(t)"> is the landmark image, <br>
<img src = "https://render.githubusercontent.com/render/math?math=i"> is the index of the video sequence in the dataset and<br>
<img src = "https://render.githubusercontent.com/render/math?math=\theta, W, w_{0}, b"> represent the learnable parameters of the discriminator network. <br>
This whole process is done by calculating a content distance term: <img src = "https://render.githubusercontent.com/render/math?math=L_{CNT}">

# Fine tuning

After this meta-learning process completes on the training data, the system can start creating video sequences from the unseen images. For that we use 
<center>  
<img src = "https://render.githubusercontent.com/render/math?math=\hat{e}_{NEW} = \frac{1}{T}\sum_{t=1}^{T}E(x(t), y(t),\phi)"><br></center>

here the value of <img src = "https://render.githubusercontent.com/render/math?math=\phi"> is reused from the meta-learning stage.

# Implementations
#### I will not bore you anymore and move straight to the implementation(s) 

Before I started trying out my own implementation, I scoured [PapersWithCode](https://www.paperswithcode.com) to see if I could find some implementations for this paper by other people and I can happily say that I actually did. I found [8 implementations](https://paperswithcode.com/paper/few-shot-adversarial-learning-of-realistic). some of them really promising. Among the 8 implementations, I found 3 worth discussing. You can check the other ones out if you want, but for this blog, I'll be focussing only on these three.

- [Realistic-Neural-Talking-Head-Models by vincent-thevenin](https://github.com/shoutOutYangJie/Few-Shot-Adversarial-Learning-for-face-swap)
- [talking-heads by grey-eye](https://github.com/grey-eye/talking-heads)
- [Few-Shot-Adversarial-Learning by shoutOutYangJie](https://github.com/shoutOutYangJie/Few-Shot-Adversarial-Learning-for-face-swap)

After going through these, I did not think that me implementing my own code would be necessary because these implementations cover almost everything along with a few **interesting** findings which I will discuss later on in this blog. Also, my machine would not be able to handle the resource requirements of this paper and by the time you are reading this blog, it would still be in training.


## Prerequisites and Dependencies
Before you go on to try out your own code or run any one of these, there are a few dependencies that you need to install.

1. [VGGFace](https://arxiv.org/pdf/1703.07332.pdf) - A very Deep convolutional Neural Network architecture:
   
   - Follow the following steps to install VGGface:
        - > wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz <br>
        - > tar xvzf vgg_face_caffe.tar.gz <br>
        - > sudo apt install caffe-cuda <br>
        - > pip install mmdnn <br>
        - > for python3 run: <br>
            > python3 -m pip install mmdnn
            
2. Libraries: Install the folllowing libraries:

    - face-alignment
        > python3 -m pip install face-alignment
    - torch 
        > python3 -m pip install torch
    - numpy
        > python3 -m pip install numpy
    - cv2 
        > python3 -m pip install opencv-python
    - matplotlib
        > python3 -m pip install matplotlib
    - tqdm
        > python3 -m pip install tqdm
        
### 3. Dataset
The model in the paper was trained on the [VoxCeleb2 Dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). The dataset is about 300GB in size and contains about 140,000 videos of talking heads.

# vincent-thevenin's implementation:
This implementation of the paper has been the best so far in my opinion. The dataset is included in the [github repo](https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models) and they have also provided [pretrained weights](https://drive.google.com/open?id=1vdFz4sh23hC_KIQGJjwbTfUdPG-aYor8) 

There are separate files for differnt implementations depending onwhat you want to do:

There is a **train.py** file which you can run to train the model yourself.

Then there is a **embedder_inference.py** file which needs a trained model (either the pretrained modelor the one you train yourself). This file gives you the embedding vector as the output.

The **fine_tuning_training.py** file is used to further fine tune the trained model. (Requires the pre trained model and the embedding vector)

If you want to try this model using your webcam, you can run the **webcam_inference.py** file.

The architecture they used has also been very well explained in the **readme.md** file on the [github repo](https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models).

# grey-eye's implementation

This implementation has the best documentation so far.  and there is also a .sh file to download VGGFace. You can download all the dependencies of this implementation from the **requirements.txt** file. All you need to do is run the following command in your terminal: 
> python3 -m pip install -r requirements.txt

grey-eye notes that in the original paper, they seem to process the videos dynamically, that is, with each iteration they select a new video. Although this has not been explicitly stated in the paper, it is also not said that they do it otherwise. Following this approach would be extremely time consuming because with each iteration you would have to pre-process the videos. This implementation gives a workaround for that (sort of) by preprocessing all the data beforehand. This gives a resulting dataset which is a lot lighter than the original dataset and easier to work with.

The architecture details can again be found in the **readme.md** file on the [github repo](https://github.com/grey-eye/talking-heads)

# shoutOutYangJie's implementation

This implementation works on the [VoxCeleb1 Dataset](www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/dense-face-frames.tar.gz) which is only 36GB in size. The most interesting thing I found about this implementation is that you can see the real-time training results in the **training visual**  directory. By using this visual implementation, shoutOutYangJie found out some problems in the Generator architecture. 

This implementation is the easiest to run and try 
1. Download the dataset and run the **get_landmarks.py** file.
2. Run the VggFace after download
3. Run the **train.py** file using 
     > python ./train.py
     
# Conclusion
You can try any of these implementations or you can try out your own implementation or do both and compare the results. All in all, this is a really interesting paper and the implementations are even more interesting. I learnt  a great deal while reading for this blog and I hope you also learnt something after reading this blog.

