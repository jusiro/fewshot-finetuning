import numpy as np
from PIL import Image
import cv2


def visualize_prediction(im, y, yhat, path_out):
    axis = {'x': -3, 'y': -2, 'z': -1}
    cmap = cv2.COLORMAP_JET

    idx = np.argwhere(y == 1)
    slice_id = int(np.mean(idx[:, -1]))

    im_sliced = im[:, :, slice_id]
    yhat_sliced = yhat[:, :, slice_id]

    xh = cv2.cvtColor(np.uint8(im_sliced * 255), cv2.COLOR_GRAY2RGB)
    heatmap_mhat = cv2.applyColorMap(np.uint8(yhat_sliced * 255), cmap)
    fin_predicted = cv2.addWeighted(xh, 0.7, heatmap_mhat, 0.3, 0)

    cv2.imwrite(path_out + '.tiff', fin_predicted)