import numpy as np
import cv2
import os


def visualize_query(args):

    # Prepare visualization folder
    path_vis = args.out_path + 'visualizations/' + args.query["name"] + "/"
    if not os.path.isdir(path_vis):
        os.makedirs(path_vis)

    # Prepare image name
    image_id = args.method + '_' + args.organ + '_k_' + str(args.k) + '_fold_' + str(args.iFold)

    # Save visualization for z = 0 - Central slice
    if not os.path.isdir(path_vis + 'z_0/'):
        os.makedirs(path_vis + 'z_0/')
    visualize_zdim(im=args.query["image"].detach().numpy()[0, :, :, :], y=args.query["label"][0, :, :, :],
                   yhat=args.query["prediction"][0, :, :, :],
                   path_out=path_vis + 'z_0/' + image_id, offset=0)
    # Save visualization for z = +10
    if not os.path.isdir(path_vis + 'z_10/'):
        os.makedirs(path_vis + 'z_10/')
    visualize_zdim(im=args.query["image"].detach().numpy()[0, :, :, :], y=args.query["label"][0, :, :, :],
                   yhat=args.query["prediction"][0, :, :, :],
                   path_out=path_vis + 'z_10/' + image_id, offset=10)
    # Save visualization for z = -10
    if not os.path.isdir(path_vis + 'z_-10/'):
        os.makedirs(path_vis + 'z_-10/')
    visualize_zdim(im=args.query["image"].detach().numpy()[0, :, :, :], y=args.query["label"][0, :, :, :],
                   yhat=args.query["prediction"][0, :, :, :],
                   path_out=path_vis + 'z_-10/' + image_id, offset=-10)


def visualize_zdim(im, y, yhat, path_out, offset=0):

    def apply_overlay(x, y, cmap):
        # Prepare overlay
        overlay = cmap[np.int32(y).flatten()]
        R, C = y.shape[:2]
        overlay = overlay.reshape((R, C, -1))

        # Apply overlay
        img_rgb = cv2.cvtColor(np.uint8(x * 255), cv2.COLOR_GRAY2RGB)  # grayscale image to rgb
        overlay = cv2.addWeighted(img_rgb, 0.4, np.uint8(overlay * 255), 0.6, 0)

        return overlay

    # Select pallete for colors
    cmap = [[  0,   0,   0],
            [235, 172,  35],
            [184,   0,  88],
            [  0, 140, 249],
            [  0, 110,   0],
            [  0, 187, 173],
            [209,  99, 230],
            [  0, 198, 248],
            [178,  69,   2],
            [255, 146, 135],
            [ 89,  84, 214],
            [135, 133,   0],
            [  0, 167, 108]],
    cmap = np.squeeze(np.array(cmap) / 255)

    # Select z slide with most annotations
    idx = np.argwhere(y > 0)
    slice_id = int(np.mean(idx[:, -1]) + offset)

    # Get image and predictions (3D->2D)
    im_sliced = im[:, :, slice_id]
    y_sliced = y[:, :, slice_id]
    yhat_sliced = yhat[:, :, slice_id]

    # Overlay image with gt and prediction
    overlay_y = apply_overlay(im_sliced, y_sliced, cmap)
    overlay_yhat = apply_overlay(im_sliced, yhat_sliced, cmap)

    # Save image and overlaid reference/prediction
    cv2.imwrite(path_out.replace(path_out.split("/")[-1], "") + 'img_' + 'z' + str(slice_id) + '.tiff',
                cv2.cvtColor(np.uint8(im_sliced * 255), cv2.COLOR_GRAY2RGB))
    cv2.imwrite(path_out.replace(path_out.split("/")[-1], "") + '_y_' + 'z' + str(slice_id) + '.tiff',
                overlay_y)
    cv2.imwrite(path_out + '_yhat_' + 'z' + str(slice_id) + '.tiff', overlay_yhat)