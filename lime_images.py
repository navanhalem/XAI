import numpy as np
import skimage
from skimage import io, transform
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.applications.imagenet_utils import decode_predictions


def transform_img_fn_ori(url):
    img = skimage.io.imread(url)
    img = skimage.transform.resize(img, (299, 299))
    img = (img - 0.5) * 2
    img = np.expand_dims(img, axis=0)
    preds = inet_model.predict(img)
    for i in decode_predictions(preds)[0]:
        print(i)
    return img


inet_model = inc_net.InceptionV3()
images_inc_im = transform_img_fn_ori('cat_image.jpg')

from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images_inc_im[0].astype('double'), inet_model.predict,
                                         top_labels=3, hide_color=0, num_samples=1000)

temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=True)
temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
ax1.imshow(mark_boundaries(temp_1, mask_1))
ax2.imshow(mark_boundaries(temp_2, mask_2))
ax1.axis('off')
ax2.axis('off')
