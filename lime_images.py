import numpy as np
import skimage
from skimage import io, transform
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.applications.imagenet_utils import decode_predictions

inet_model = inc_net.InceptionV3()
img = skimage.io.imread('cat_image.jpg')
img = skimage.transform.resize(img, (299, 299))
img = (img - 0.5) * 2
img = np.expand_dims(img, axis=0)
preds = inet_model.predict(img)
for i in decode_predictions(preds)[0]:
    print(i)

from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(img[0].astype('double'), inet_model.predict,
                                         top_labels=3, hide_color=0, num_samples=500)

temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=True)
temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
temp_3, mask_3 = explanation.get_image_and_mask(explanation.top_labels[1], positive_only=False, num_features=10,
                                                hide_rest=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
ax1.imshow(img[0])
ax2.imshow(mark_boundaries(temp_2, mask_2))
ax3.imshow(mark_boundaries(temp_3, mask_3))
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
