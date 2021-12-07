<head><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.css" integrity="sha384-yFRtMMDnQtDRO8rLpMIKrtPCD5jdktao2TV19YiZYWMDkUR5GQZR/NOVTdquEx1j" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/katex.min.js" integrity="sha384-9Nhn55MVVN0/4OFx7EE5kpFBPsEMZxKTCnA+4fqDmg12eCTqGi6+BB2LjY8brQxJ" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script></head>

<h1> Introduction </h1>
This project is based on the paper <a href="https://cseweb.ucsd.edu//~viscomp/classes/cse274/fa21/readings/a193-kalantari.pdf">Learning-Based View Synthesis for Light Field Cameras</a> by Kalantari et al.  In this paper, the authors use a 2-stage neural network to synthesize images from non-input view directions in a light field camera.  In theory, this work could be used to increase the spatial resolution of consumer light field cameras while retaining the same angular resolution, by simply synthesizing the intermediate views.

<h1> Overview of the Original Paper </h1>
Below, we give a brief synopsis of how this paper works.

<h2> Architecture </h2>
The neural network architecture in this paper consists of two CNNs, each with 4 convolutional layers and no fully-connected layers.  The network takes in the 4 corner views of an 8x8 angular resolution grid, as well as the (u, v) coordinates of the desired input view.  The full network learns to directly synthesize an image from the desired input view.

<h3> Disparity Estimator </h3>
The first CNN is called a "disparity estimator", and it aims to roughly estimate the disparity (a measure of the "motion" or "distance" between pixels in two camera views) as a 1-channel image.  Its input is 200 image features which we manually create before applying the network.  These features are intended to help the network see the disparity at each point in the image.  First, we <i>backward warp</i> the 4 input images (converted to grayscale); this step shifts each of the pixels in each input image by a predefined disparity level $'d_l'$.  Thus we find a backwarped image $'\bar{L}_{p_i}^{d_l}(s) = L_{p_i}\[s + (p_i - q)d_l\]'$, where $'p_i'$ and $'q'$ are the (u, v) coordinates of the input and target views, respectively.  We perform this for each of the 4 input views, and warp using 100 predefined disparity levels between -21 and 21.  Then, 100 features are found by averaging over the 4 views at each disparity level, and 100 features are found by taking the standard deviation of the 4 views at each disparity level.  We stack the 200 features into a 100 x h x w tensor.

<img/>

<h3> Color Predictor </h3>
The color predictor network uses the disparity estimator output, the 4 input images, and the (u, v) coordinates of the target view to determine the 3-channel output view.  To create the features for this network, we use the disparity estimator output to <i>forward warp</i> the 4 (color) input views.  This is given by the equation $'\bar{L}_{p_i}(s) = L_{p_i}\[s + (p_i - q)D_q(s)\]'$, where $'D_q(s)'$ gives the disparity estimator's output at pixel $'s'$.  The warped views are then concatenated with the (u, v) target view coordinates as well as the disparity output, giving a 3*4 + 2 + 1 = 15 channel input.  The layers in this step are the same as in the disparity estimator, but the input layer takes in 15 channels rather than 200.

<h2> Other Paper Details </h2>
In the original paper, training occurs on 60x60 patches, with batch size 20.  They use the L2 loss, and use ADAM to update the weights of the network.  Additionally, since the warp operations will likely try to read values between pixels if implemented naively, they use bicubic interpolation to sample the all images that are warped (forward or backward).  Backpropagation through the bicubic interpolation step during the forward warp is performed numerically.

<h1> My Implementation </h1>
I implemented this paper for my final project in CSE 274 Fall 2021: "Sampling and Reconstruction of Visual Appearance: From Denoising to View Synthesis".  Here, I detail my implementation and how I got there.
<h2> Early Steps </h2>
Earlier in the quarter, I was interested in implementing <a href=<https://cseweb.ucsd.edu//~viscomp/classes/cse274/fa21/papers/nex-cvpr21.pdf">NeX: Real-time View Synthesis with Neural Basis Expansion </a> (Wizadwongsa et al. 2021).  However, after working on the implementation for a while, I realised that given my resources and time, it would be too complicated for me to implement and run in a single quarter.  I started looking at the other papers covered in this course, and briefly played with the idea of implementing one of the denoising papers, but decided to instead implement this learning-based view synthesis paper from 2016, as it fell within my interests and matched the resources available to me.  With that, I began implementing the network and finding solutions that allowed me to access school GPUs.

<h2> Frameworks / Resource Acquisition </h2>
Our course gives access to UCSD DataHub, a service which allows students to connect to computers with decent GPUs and pre-installed machine learning environments.  Through this service, I was able to use a 2080 Ti, accessing it through an online Jupyter Notebook.  I used Pytorch (with CUDA) for implementation and training of the network, as well as numpy for a few transformations of the data that aren't required to be differentiable.  I also used cv2 (OpenCV) for writing my video results.  It took quite a while, but I was also able to convince the kind folks at DataHub to place my ~30gb worth of data on their computers, unzip the folders, and give me access to them.

<h2> Differences from the Original Paper </h2>
My implementation differed from the original paper slightly, as it used different frameworks and had access to different resources.  I summarize the differences below:
<ul>
  <li>In the original paper (as far as I could tell), patches are generated before training and inserted into local folders.  This allows the batches to be less correlated -- a single batch can have patches from many images with many input views.  However, due to the limitations of DataHub, I found that this would be infeasible.  As a student, I only have access to ~10gb of local storage, and ~16gb of RAM, so it would neither be possible to push all the patches to new files nor to construct uncorrelated batches from multiple images and views.  Thus, in my implementation, I load a single light field image at a time and train on patches exclusively from that image with a single target view.  This is a significant batch correlation, which likely decreased my convergence rate, but I hope I can convince you that I still generate decent results.</li>
  <li>I used Pytorch (with CUDA) for implementation and training, whereas the original paper authors used MATLAB with MatConvNet</li>
  <li>I used uniform He initialization (assuming ReLU activations) rather than Xavier initialization (which the authors use).  I made this choice after some short research that informed me that Xavier is optimized for sigmoid activations rather than ReLU activations.</li>
  <li>I likely trained the network for less time.  Although I used the DataHub GPU for ~2 days, it seems that training is slower during peak hours.  This means that my access to the GPU is mediated by other students' accessing it as well, and thus I likely did not train for the same effective GPU time as in the paper.</li>
</ul>

<h2> Training </h2>
I trained for a few days, until the network seemed to have reached its lowest test loss.  The train/test loss graphs are shown below.  Note that the scales are different -- for the training loss, I did not normalize by the number of samples, but the curve is nonetheless clear.

<h2> Other Implementation Details </h2>
In the results I presented in class, I mentioned that I was having issues with low saturation in my output images.  After a student commented that this could likely be a bug related to gamma correction, I looked at the paper authors' code to find out what type of tone correction I would have to apply.  The result is that my results were correct in my presentation, but needed to be gamma corrected by 1.5 and also saturation-boosted by 1.5.  Thus my images in this report are more color-correct and more highly-saturated than those from my presentation.  In addition, the images shown below are from versions of the network trained further than the images in the presentation, and thus there will likely be fewer artifacts overall.

For the forward and backward image warps, I utilized Pytorch's built-in meshgrid and grid_sample functions.  This allowed me to create a uniform grid of (x, y) points, add the necessary offsets to each coordinate based on the type of warp, and then sample the bicubic interpolated input images using the determined grid.  Additionally, since both of these are Pytorch functions, I was able to use Pytorch's default gradients for each of these functions, thus abstracting away the messy technical details involved in finding numerical gradients for bicubic interpolation.

<h1> Results </h1>
Finally, we examine the results.  First, here are the videos of my network cycling through every synthesized input view on the original Flower1.png light field image.

<!-- <video width="541" height="376" controls>
  <source src="movie.mp4" type="video/mp4"/> 
Your browser does not support the video tag.
</video> -->

We also show results circling around the edges for several more light fields (Flower1.png, Seahorse.png).

<!-- <video width="541" height="376" controls>
  <source src="movie.mp4" type="video/mp4"/> 
Your browser does not support the video tag.
</video> -->

<!-- <video width="541" height="376" controls>
  <source src="movie.mp4" type="video/mp4"/> 
Your browser does not support the video tag.
</video> -->

In the original paper, there are also decent results for extrapolated views, though these tend to have lots of artifacts around occlusion boundaries.  Here, I show some extrapolated results for my implementation of the network.

<!-- <video width="541" height="376" controls>
  <source src="movie.mp4" type="video/mp4"/> 
Your browser does not support the video tag.
</video> -->

<!-- <video width="541" height="376" controls>
  <source src="movie.mp4" type="video/mp4"/> 
Your browser does not support the video tag.
</video> -->

Finally, I show results for images taken on a cellphone.  Since the original light field camera has a very small baseline and the views are very well-calibrated, it is clear that using cellphone images will not get perfect results on a network trained on the light field images.  However, the results still retain some quailty, and it is nonetheless interesting that one can synthesize these views at this quality from cellphone images.

<!-- <video width="541" height="376" controls>
  <source src="movie.mp4" type="video/mp4"/> 
Your browser does not support the video tag.
</video> -->

<h2> Conclusion </h2>
