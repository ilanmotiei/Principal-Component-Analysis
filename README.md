# Principal-Component-Analysis
A python implementation of the PCA method, which can be used in ML for denoising data.
Photos used as a DB for this task are of 'Arial Sharon', taken from the 'lfw_people' data base.
The code shows at first some of the 'eigen-faces' we get when applying the method on the photos.
Then it shows several examples of applying a compression on the photos and then reconstructing them (using PCA) (in the plots 'k' is the reduction dimension).
Last it shows the upper-bound (factored by some constant) on the distance of the reconstructed photos from the original ones, as a function of k, 
allowing us to choose the apropriate 'k' (reduction dimension) for denoising the data.
