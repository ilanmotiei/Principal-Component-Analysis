import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np


def plot_vector_as_image(image, h, w, title=""):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimesnions of original picture
    """
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title(title, size=12)
    plt.show()


def get_pictures_by_name(name='Ariel Sharon', min_faces_per_person=70):
    """
    Given a name returns all the pictures of the person with this specific name.
    """

    lfw_people = load_data(min_faces_per_person)
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if target == target_label:
            image_vector = image.reshape((h*w, ))
            selected_images.append(image_vector)

    selected_images = np.array(selected_images)
    return selected_images, h, w


def load_data(min_faces_per_person=70):
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, resize=0.4)
    return lfw_people


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return. assuming k < d;

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest
      k eigenvectors of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    n = X.shape[0]; d = X.shape[1]

    normalized_X = X - np.mean(X, axis=0, keepdims=True)
    normalized_X = X

    sigma = np.matmul(np.transpose(normalized_X), normalized_X) / n

    eigenValues, eigenVectors = np.linalg.eig(sigma);

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx];
    eigenVectors = eigenVectors[:,idx];

    S = eigenValues[:k]
    U = eigenVectors[:, :k]
    # print(U.shape)

    return U.T, S


def EigenFaces(name="Ariel Sharon"):
    Data, h, w = get_pictures_by_name(name=name, min_faces_per_person=50)

    PCA_eigenVectors = PCA(Data, k=1850)[0]
    #print(PCA_eigenVectors.shape)

    for i in range(0, 10):
        plot_vector_as_image(PCA_eigenVectors[i], h, w)


def CompressionExample(name="Ariel Sharon"):
    Data, h, w = get_pictures_by_name(name=name, min_faces_per_person=50)
    n = Data.shape[0]; d = Data.shape[1]

    Ks = [1, 5, 10, 30, 50, 100, 200, 1850]
    total_distances = [0.1] * len(Ks)
    data_Collection = list()
    distances = list()


    for k in Ks:
        encoder = PCA(Data, k)[0]
        decoder = np.transpose(encoder)

        transformed_data = np.ndarray(shape=(n, d))

        for i in range(0, n):
            transformed_data[i] = np.transpose(np.matmul(decoder, np.matmul(encoder, Data[i])))

        transformed_data += transformed_data.mean(axis=0, keepdims=True)

        data_Collection.append(transformed_data)
        distances.append(np.sum(np.subtract(Data, transformed_data) ** 2))

    indexes = np.random.choice(n, size=5)

    for idx in indexes:
        plot_vector_as_image(Data[idx], h, w, title="Original Image (50x37 pixels)")

        for j in range(0, len(Ks)):
            k = Ks[j]
            plot_vector_as_image(data_Collection[j][idx], h, w, title=("PCAed Image : K = " + str(k)))

    plt.plot(Ks, distances)
    plt.xlabel("K")
    plt.ylabel("Total Penalty (Distances)")
    plt.title("Distances / K")

    plt.show()


def Loss_as_a_function_of_k(name="Ariel Sharon"):
    X, h, w = get_pictures_by_name(name=name, min_faces_per_person=50)
    n = X.shape[0]; d = X.shape[1]

    X = X - X.mean(axis=0, keepdims=True)
    sigma = 1/n * np.dot(np.transpose(X), X)

    eigenValues, eigenVectors = np.linalg.eig(sigma);

    idx = eigenValues.argsort()[::-1];
    eigenValues = eigenValues[idx];

    Ks = list(range(1, d+1))
    Distance_Penalty = list()
    Distance_Penalty.append(sum(eigenValues))

    for i in range(1, d+1):
        Distance_Penalty.append(Distance_Penalty[i-1] - eigenValues[i-1])

    del Distance_Penalty[0]

    plt.plot(Ks, Distance_Penalty)
    plt.xlabel("K")
    plt.ylabel("Penalty (Loss)")
    plt.title("Linear PCA Loss at Reconstruction / K ")

    plt.show()


# ---------------------------------------------- Running Session ----------------------------------------------- #

if __name__ == "__main__":

    EigenFaces()
    CompressionExample()
    Loss_as_a_function_of_k()









