import numpy as np
import skimage
from skimage import io, transform
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt

# Load the inception V3 model
inet_model = inc_net.InceptionV3()

# Read the image and transform it into an image that can be read by the inception V3 model
image = skimage.io.imread('cat_image.jpg')
image = skimage.transform.resize(image, (299, 299))
image = (image - 0.5) * 2
image = np.expand_dims(image, axis=0)

# Make a prediction for the image
prediction = inet_model.predict(image)
for i in decode_predictions(prediction)[0]:
    print(i)

# Create an explainer object and generate an explanation
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image[0].astype('double'), inet_model.predict,
                                         top_labels=3, hide_color=0, num_samples=500)

temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
temp_3, mask_3 = explanation.get_image_and_mask(explanation.top_labels[1], positive_only=False, num_features=10,
                                                hide_rest=False)

# plot the original image and the (area's that contributed to the) predictions
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
ax1.imshow(image[0])
ax2.imshow(mark_boundaries(temp_2, mask_2))
ax3.imshow(mark_boundaries(temp_3, mask_3))
ax1.set_title('original image')
ax2.set_title(f'prediction: {decode_predictions(prediction)[0][0][1]}')
ax3.set_title(f'prediction: {decode_predictions(prediction)[0][1][1]}')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
