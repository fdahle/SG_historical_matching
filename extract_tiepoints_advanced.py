import copy
import cv2
import numpy as np
import torch
import math

from SuperGlue.matching import Matching

import display_tiepoints as dt

debug_show_resized_tiepoints = False
debug_show_final_tiepoints = False
debug_show_original_tiepoints_subset = False
debug_show_new_tiepoints_subset = False
debug_show_additional_tiepoints_subset = False


def extract_tiepoints_advanced(input_img1, input_img2, mask_1=None, mask_2=None,
                            max_width=2000, max_height=1500,
                            extra_mask_padding=0,
                            additional_matching=False,
                            min_threshold=None,
                            filter_outliers=True,
                            keep_resized_points=False,
                            verbose=False):

    """
    extract_tiepoints_advanced:
    This function extracts tie-points between two images making use of SuperGlue. First an initial matching on
    resized imagery is done. Afterwards, subsets are extracted around the location of the tie-points found during
    resizing. A second matching is then done using these subsets in original resolution. The size of these subsets
    is calculated automatically, depending on the available memory in the GPU.
    Additional matching: After tie-points are found, the transformation matrix between the two images is calculated.
    This is used, to check where a subset of one image is located in the other image. Between these subsets,
    additional matching is done and added to the final results.
    Args:
        input_img1 (np-array): One image used for tie-point matching.
        input_img2 (np-array): Second image used for tie-point matching.
        mask_1 (np-array): Mask for input_img1, binary np array (0 means points at that location are filtered).
        mask_2 (np-array): Mask for input_img2, binary np array (0 means points at that location are filtered).
        max_width (int, 2000): width for the resized image and the initial subset
        max_height (int, 1500): height for the resized image and the initial subset
        extra_mask_padding (int, 0): The number of pixels that are subtracted at the edge of the mask
                                    (When using image enhancements, there are many false tie-points at the edges)
        additional_matching (bool, False): If true, an additional matching is done (see description at the beginning)
        min_threshold (float, None): The minimum quality required for the tie-points (Value between 0 and 1). If none,
                                     all tie-points are saved regardless of the quality.
        filter_outliers (Bool, True): If true, RANSAC is applied to filter outliers of the tie-points.
        keep_resized_points (Bool, False): If true, the tie-points find during matching of the resized images are kept
                                           as well.
        verbose (Boolean, False): If true, the status of the operations are printed
    Returns:
        all_tiepoints (np-array): Array with all the tie-points (X1, Y1, X2, Y2)
        all_confidences (list): List of confidence values of the tie_points
    """

    # check the inputs
    if mask_1 is not None:
        assert input_img1.shape == mask_1.shape
    if mask_2 is not None:
        assert input_img1.shape == mask_2.shape
    assert extra_mask_padding >= 0
    assert max_width >= 0
    assert max_height >= 0

    # function to resize the images
    def resize_img(img, height_max, width_max):

        # check if we need to resize the images
        if img.shape[0] > height_max or img.shape[1] > width_max:

            # set resized flag
            bool_resized = True

            # check if we need to resize due to width or height
            if img.shape[0] >= img.shape[1]:
                resize_factor = height_max / img.shape[0]
            else:
                resize_factor = width_max / img.shape[1]

            # get the new image shape and resize
            new_image_shape = (int(img.shape[0] * resize_factor), int(img.shape[1] * resize_factor))
            img_resized = cv2.resize(img, (new_image_shape[1], new_image_shape[0]), interpolation=cv2.INTER_NEAREST)

        else:
            # set resized flag
            bool_resized = False

            # we don't need to change the image
            img_resized = img
            resize_factor = 1

        return bool_resized, img_resized, resize_factor

    # function to actual get tie points
    def apply_sg(input_img_1, input_img_2):

        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # put stuff to cuda
        sg_img_1 = copy.deepcopy(input_img_1)
        sg_img_2 = copy.deepcopy(input_img_2)

        sg_img_1 = torch.from_numpy(sg_img_1)[None][None] / 255.
        sg_img_2 = torch.from_numpy(sg_img_2)[None][None] / 255.

        sg_img_1 = sg_img_1.to(device)
        sg_img_2 = sg_img_2.to(device)

        # superglue settings
        nms_radius = 3
        keypoint_threshold = 0.005
        max_keypoints = -1  # -1 keep all
        weights = "outdoor"  # can be indoor or outdoor
        sinkhorn_iterations = 20
        match_threshold = 0.2

        # set config for superglue
        superglue_config = {
            'superpoint': {
                'nms_radius': nms_radius,
                'keypoint_threshold': keypoint_threshold,
                'max_keypoints': max_keypoints
            },
            'superglue': {
                'weights': weights,
                'sinkhorn_iterations': sinkhorn_iterations,
                'match_threshold': match_threshold,
            }
        }

        # init the matcher
        matching = Matching(superglue_config).eval().to(device)

        # what do we want to detect
        keys = ['keypoints', 'scores', 'descriptors']

        last_data = matching.superpoint({'image': sg_img_1})
        last_data = {k + '0': last_data[k] for k in keys}
        last_data["image0"] = sg_img_1

        pred = matching({**last_data, 'image1': sg_img_2})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches_superglue = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].detach().cpu().numpy()

        # keep the matching key points
        valid = matches_superglue > -1
        mkpts_0 = kpts0[valid]
        mkpts_1 = kpts1[matches_superglue[valid]]
        m_conf = confidence[valid]

        return mkpts_0, mkpts_1, m_conf

    # function to filter tie points with masks
    def filter_with_mask(f_mktps, f_conf, f_mask1, f_mask2):

        # init array where we store the points we want to filter
        filter_values_1, filter_values_2 = [], []

        for row in f_mktps:
            if f_mask1 is not None:

                if extra_mask_padding == 0:
                    filter_values_1.append(f_mask1[row[1], row[0]])
                else:

                    min_subset_x = row[0] - extra_mask_padding
                    max_subset_x = row[0] + extra_mask_padding
                    min_subset_y = row[1] - extra_mask_padding
                    max_subset_y = row[1] + extra_mask_padding

                    if min_subset_x < 0:
                        min_subset_x = 0
                    if min_subset_y < 0:
                        min_subset_y = 0
                    if max_subset_x > f_mask1.shape[1]:
                        max_subset_x = f_mask1.shape[1]
                    if max_subset_y > f_mask1.shape[0]:
                        max_subset_y = f_mask1.shape[0]

                    mask_subset = f_mask1[min_subset_y:max_subset_y,min_subset_x:max_subset_x]

                    if 0 in mask_subset:
                        filter_values_1.append(0)
                    else:
                        filter_values_1.append(1)

            if f_mask2 is not None:

                if extra_mask_padding == 0:
                    filter_values_2.append(f_mask2[row[3], row[2]])
                else:

                    min_subset_x = row[3] - extra_mask_padding
                    max_subset_x = row[3] + extra_mask_padding
                    min_subset_y = row[2] - extra_mask_padding
                    max_subset_y = row[2] + extra_mask_padding

                    if min_subset_x < 0:
                        min_subset_x = 0
                    if min_subset_y < 0:
                        min_subset_y = 0
                    if max_subset_x > f_mask2.shape[1]:
                        max_subset_x = f_mask2.shape[1]
                    if max_subset_y > f_mask2.shape[0]:
                        max_subset_y = f_mask2.shape[0]

                    mask_subset = f_mask2[min_subset_y:max_subset_y,min_subset_x:max_subset_x]

                    if 0 in mask_subset:
                        filter_values_2.append(0)
                    else:
                        filter_values_2.append(1)

        # convert to np array
        filter_values_1 = np.asarray(filter_values_1)
        filter_values_2 = np.asarray(filter_values_2)

        filter_indices = np.logical_or(filter_values_1 == 0, filter_values_2 == 0)
        filter_indices = 1 - filter_indices
        filter_indices = filter_indices.astype(bool)

        f_mkpts = f_mktps[filter_indices,:]
        f_conf = f_conf[filter_indices]

        return f_mkpts, f_conf

    # deep copy to not change the original
    img_1 = copy.deepcopy(input_img1)
    img_2 = copy.deepcopy(input_img2)

    # images must be in grayscale
    if len(img_1.shape) == 3:
        img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    if len(img_2.shape) == 3:
        img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # check and resize the images
    bool_resized_1, img_1_resized, resize_factor_1 = resize_img(img_1, max_height, max_width)
    bool_resized_2, img_2_resized, resize_factor_2 = resize_img(img_2, max_height, max_width)

    # we try this as long as we get to results
    while True:
        try:

            # get the first batch of tie points for a smaller image
            mkpts1, mkpts2, mconf = apply_sg(img_1_resized, img_2_resized)

            # that means we didn't raise an error but found a solution
            break

        except (Exception,) as e:
            if "out of memory" in str(e):

                # free the gpu
                torch.cuda.empty_cache()

                # calculate new height and width
                max_width = int(0.9*max_width)
                max_height = int(0.9*max_height)

                # resize images again
                bool_resized_1, img_1_resized, resize_factor_1 = resize_img(img_1, max_height, max_width)
                bool_resized_2, img_2_resized, resize_factor_2 = resize_img(img_2, max_height, max_width)

                if verbose:
                    print(f"Out of memory, try again with new image size for "
                          f"img1 ({img_1_resized.shape}) and img2 ({img_2_resized.shape})")
            else:
                raise e

    if verbose:
        print(f"initial matching done ({mkpts1.shape[0]} tie-points found)")

    # merge the tiepoints of let and right image and fit it to the big image again
    mkpts = np.concatenate((mkpts1, mkpts2), axis=1)
    mkpts[:, 0] = mkpts[:, 0] * 1 / resize_factor_1
    mkpts[:, 1] = mkpts[:, 1] * 1 / resize_factor_1
    mkpts[:, 2] = mkpts[:, 2] * 1 / resize_factor_2
    mkpts[:, 3] = mkpts[:, 3] * 1 / resize_factor_2
    mkpts = mkpts.astype(int)

    # mask the points if necessary
    mkpts, mconf = filter_with_mask(mkpts, mconf, mask_1, mask_2)

    # if images were resized we need to do some more stuff to find good tie points
    if bool_resized_1 is True or bool_resized_2 is True:

        if debug_show_resized_tiepoints:
            dt.display_tiepoints(input_img1, input_img2, mkpts, mconf,
                                 title=f"resized tiepoints {mkpts.shape[0]}")

        # we need to iterate over the bigger image
        if img_1.shape[0] * img_1.shape[1] >= img_2.shape[0] * img_2.shape[1]:
            base_img = img_1
            other_img = img_2

            # the tie points must be resized again
            mkpts_base = mkpts[:, 0:2]
            mkpts_other = mkpts[:, 2:4]
        else:
            base_img = img_2
            other_img = img_1
            mkpts_base = mkpts[:, 2:3]
            mkpts_other = mkpts[:, 0:1]

        # how often does the maximum possible image fit in our base img
        num_x = math.ceil(base_img.shape[1] / max_width)
        num_y = math.ceil(base_img.shape[0] / max_height)

        # how much do we have too much
        too_much_x = num_x * max_width - base_img.shape[1]
        too_much_y = num_y * max_height - base_img.shape[0]

        # how much we need to shift the image every step
        reduce_x = int(too_much_x / (num_x - 1))
        reduce_y = int(too_much_y / (num_y - 1))

        # here we save all tiepoints
        all_tiepoints = []
        all_confidences = []

        total_number_of_tiles = math.ceil(num_y) * math.ceil(num_x)
        total_counter = 0

        # iterate over the tiles
        for y_counter in range(0, math.ceil(num_y)):

            for x_counter in range(0, math.ceil(num_x)):

                total_counter = total_counter + 1
                if verbose:
                    print(f"match {total_counter} of {total_number_of_tiles}:")

                # calculate the extent of the tile
                min_x = x_counter * max_width - reduce_x * x_counter
                max_x = (x_counter + 1) * max_width - reduce_x * x_counter
                min_y = y_counter * max_height - reduce_y * y_counter
                max_y = (y_counter + 1) * max_height - reduce_y * y_counter

                # get the indices of the points that are in this tile
                base_indices = np.where((mkpts_base[:, 0] >= min_x) & (mkpts_base[:, 0] <= max_x) &
                                        (mkpts_base[:, 1] >= min_y) & (mkpts_base[:, 1] <= max_y))

                # based on these indices get the points of the "other" image
                other_points = mkpts_other[base_indices]

                # if there are no tie points we can continue
                if other_points.shape[0] == 0:
                    if verbose:
                        print("  No tie-points to match")
                    continue

                # init step for the percentile loop
                step = 0

                # get tie points based on percentiles, do it so long until we found a good image
                while True:

                    # calculate the percentile of these selected points
                    other_percentile_sm = np.percentile(other_points, step, axis=0).astype(int)
                    other_percentile_la = np.percentile(other_points, 100-step, axis=0).astype(int)

                    # calculate extent of the subset (min_x, max_y, min_y, max_y
                    min_subset_x = other_percentile_sm[0]  # - int(max_width / 2)
                    max_subset_x = other_percentile_la[0]  # + int(max_width / 2)
                    min_subset_y = other_percentile_sm[1]  # - int(max_height / 2)
                    max_subset_y = other_percentile_la[1]  # + int(max_height / 2)

                    # if extent is smaller than what we could -> enlarge
                    if max_subset_x - min_subset_x < max_width:
                        missing_width = max_width - (max_subset_x - min_subset_x)
                        min_subset_x = min_subset_x - int(missing_width/2)
                        max_subset_x = max_subset_x + int(missing_width/2)
                    if max_subset_y - min_subset_y < max_height:
                        missing_height = max_height - (max_subset_y - min_subset_y)
                        min_subset_y = min_subset_y - int(missing_height/2)
                        max_subset_y = max_subset_y + int(missing_height/2)

                    # make absolutely sure that we are not getting out of image bounds
                    # should not be necessary, but better safe than sorry
                    if min_subset_x < 0:
                        min_subset_x = 0
                    if min_subset_y < 0:
                        min_subset_y = 0
                    if max_subset_x > other_img.shape[1]:
                        max_subset_x = other_img.shape[1]
                    if max_subset_y > other_img.shape[0]:
                        max_subset_y = other_img.shape[0]

                    # the tie points we found are good, because the subset is small enough!
                    if max_subset_x - min_subset_x <= max_width and \
                       max_subset_y - min_subset_y <= max_height:
                        break

                    # with the current step we found too many tie-points
                    # -> decrease the range in which we look for tie points
                    step = step + 5

                # get the subsets for superglue extraction
                base_subset = base_img[min_y:max_y, min_x:max_x]
                other_subset = other_img[min_subset_y:max_subset_y, min_subset_x:max_subset_x]

                if debug_show_original_tiepoints_subset:
                    # create temp mktps to show in display_tiepoints
                    temp_mkpts = np.concatenate((mkpts_base[base_indices], mkpts_other[base_indices]), axis=1)
                    temp_mkpts = copy.deepcopy(temp_mkpts)
                    temp_conf = mconf[base_indices]

                    # adapt the tie points of the other image to account that we look at a subset
                    temp_mkpts[:, 0] = temp_mkpts[:, 0] - min_x
                    temp_mkpts[:, 1] = temp_mkpts[:, 1] - min_y
                    temp_mkpts[:, 2] = temp_mkpts[:, 2] - min_subset_x
                    temp_mkpts[:, 3] = temp_mkpts[:, 3] - min_subset_y

                    dt.display_tiepoints(base_subset, other_subset, points=temp_mkpts, confidence=temp_conf,
                                         title=f"original tiepoints for subset ({y_counter}/{x_counter})")

                sub_resize_factor_1, sub_resize_factor_2 = 1, 1

                # get the tie points for these subsets
                while True:
                    try:
                        mkpts1_sub, mkpts2_sub, mconf_sub = apply_sg(base_subset, other_subset)

                        # that means we didn't raise an error but found a solution
                        break

                    except (Exception,) as e:
                        if "out of memory" in str(e):

                            # free the gpu
                            torch.cuda.empty_cache()

                            # calculate new height and width
                            new_width = int(0.9 * base_subset.shape[1])
                            new_height = int(0.9 * base_subset.shape[0])

                            # resize images again
                            _, base_subset, sub_resize_factor_1 = resize_img(base_subset, new_height, new_width)
                            _, other_subset, sub_resize_factor_2 = resize_img(other_subset, new_height, new_width)

                            if verbose:
                                print(f"Out of memory, try again with new image size for "
                                      f"subset of img1 ({base_subset.shape}) and subset of img2 ({other_subset.shape})")
                        else:
                            raise e

                mkpts_sub = np.concatenate((mkpts1_sub, mkpts2_sub), axis=1)

                if debug_show_new_tiepoints_subset:
                    dt.display_tiepoints(base_subset, other_subset, mkpts_sub, mconf_sub,
                                         title=f"new tiepoints for subset ({y_counter}/{x_counter})")

                mkpts_sub[:, 0] = mkpts_sub[:, 0] * 1 / sub_resize_factor_1
                mkpts_sub[:, 1] = mkpts_sub[:, 1] * 1 / sub_resize_factor_1
                mkpts_sub[:, 2] = mkpts_sub[:, 2] * 1 / sub_resize_factor_2
                mkpts_sub[:, 3] = mkpts_sub[:, 3] * 1 / sub_resize_factor_2
                mkpts = mkpts.astype(int)

                # adapt the tie points of the other image to account that we look at a subset
                mkpts_sub[:, 0] = mkpts_sub[:, 0] + min_x
                mkpts_sub[:, 1] = mkpts_sub[:, 1] + min_y
                mkpts_sub[:, 2] = mkpts_sub[:, 2] + min_subset_x
                mkpts_sub[:, 3] = mkpts_sub[:, 3] + min_subset_y

                # add the tie points and confidence values to the list
                all_tiepoints.append(mkpts_sub)
                for elem in mconf_sub:
                    all_confidences.append(elem)

                if verbose:
                    print(f"  {mkpts_sub.shape[0]} tie-points matched.")

        if len(all_tiepoints)==0:
            return None, None

        # stack tie points together (optional with the resized)
        all_tiepoints = np.vstack(all_tiepoints)
        if keep_resized_points:
            all_tiepoints = np.concatenate((all_tiepoints, mkpts), axis=0)

        # stack all confidences (optional with the resized)
        all_confidences = np.array(all_confidences)
        if keep_resized_points:
            all_confidences = np.concatenate((all_confidences, mconf), axis=0)

        # remove duplicates
        all_tiepoints, unique_indices = np.unique(all_tiepoints, return_index=True, axis=0)
        all_tiepoints = all_tiepoints.astype(int)
        all_confidences = all_confidences[unique_indices]

        all_tiepoints, all_confidences = filter_with_mask(all_tiepoints, all_confidences, mask_1, mask_2)

    else:
        all_tiepoints = mkpts
        all_confidences = mconf

    # apply threshold
    if min_threshold is not None:

        num_tp_before = all_tiepoints.shape[0]
        all_tiepoints = all_tiepoints[all_confidences >= min_threshold]
        all_confidences = all_confidences[all_confidences >= min_threshold]
        num_tp_after = all_tiepoints.shape[0]

        if verbose:
            print(f"{num_tp_before - num_tp_after} tiepoints removed with a quality lower than {min_threshold}.")

    # filter for outliers
    if filter_outliers and all_tiepoints.shape[0] >= 4:
        _, mask = cv2.findHomography(all_tiepoints[:, 0:2], all_tiepoints[:, 2:4], cv2.RANSAC, 5.0)
        mask = mask.flatten()

        if verbose:
            print(f"{np.count_nonzero(mask)} outliers removed.")

        # 1 means outlier
        all_tiepoints = all_tiepoints[mask == 0]
        all_confidences = all_confidences[mask == 0]

    if additional_matching:
        # get affine transformation
        from skimage import transform as tf

        trans_mat = tf.estimate_transform('affine', all_tiepoints[:,0:2], all_tiepoints[:,2:4])
        trans_mat = np.array(trans_mat)[0:2, :]

        # here we save all tiepoints
        additional_tiepoints = []
        additional_confidences = []

        # iterate over tiles again
        total_counter = 0
        for y_counter in range(0, math.ceil(num_y)):
            for x_counter in range(0, math.ceil(num_x)):

                total_counter = total_counter + 1
                if verbose:
                    print(f"match {total_counter} of {total_number_of_tiles}:")

                # calculate the extent of the tile
                min_x = x_counter * max_width - reduce_x * x_counter
                max_x = (x_counter + 1) * max_width - reduce_x * x_counter
                min_y = y_counter * max_height - reduce_y * y_counter
                max_y = (y_counter + 1) * max_height - reduce_y * y_counter

                # create points from extent
                extent = np.asarray([
                    [min_x, min_y],
                    [max_x, max_y],
                ])

                # resample points
                extent[:, 0] = trans_mat[0][0] * extent[:, 0] + trans_mat[0][1] * extent[:, 1] + trans_mat[0][2]
                extent[:, 1] = trans_mat[1][0] * extent[:, 0] + trans_mat[1][1] * extent[:, 1] + trans_mat[1][2]

                # get min and max values
                res_min_x = extent[0][0]
                res_max_x = extent[1][0]
                res_min_y = extent[0][1]
                res_max_y = extent[1][1]

                # some images are out of range
                if res_min_x < 0 or res_max_x < 0 or res_min_y < 0 or res_max_y < 0:
                    if verbose:
                        print("skip because out of range")
                    continue
                if res_min_x > other_img.shape[1] or res_max_x > other_img.shape[1] or \
                        res_min_y > other_img.shape[0] or res_max_y > other_img.shape[0]:
                    if verbose:
                        print("skip because out of range")
                    continue

                base_subset = base_img[min_y:max_y, min_x:max_x]
                other_subset = other_img[res_min_y:res_max_y, res_min_x:res_max_x]

                sub_resize_factor_1, sub_resize_factor_2 = 1, 1

                # get the tie points for these subsets
                while True:
                    try:
                        mkpts1_sub, mkpts2_sub, mconf_sub = apply_sg(base_subset, other_subset)

                        # that means we didn't raise an error but found a solution
                        break

                    except (Exception,) as e:
                        if "out of memory" in str(e):

                            # free the gpu
                            torch.cuda.empty_cache()

                            # calculate new height and width
                            new_width = int(0.9 * base_subset.shape[1])
                            new_height = int(0.9 * base_subset.shape[0])

                            # resize images again
                            _, base_subset, sub_resize_factor_1 = resize_img(base_subset, new_height, new_width)
                            _, other_subset, sub_resize_factor_2 = resize_img(other_subset, new_height, new_width)

                            if verbose:
                                print(f"Out of memory, try again with new image size for "
                                      f"subset of img1 ({base_subset.shape}) and subset of img2 ({other_subset.shape})")
                        else:
                            raise e

                mkpts_sub = np.concatenate((mkpts1_sub, mkpts2_sub), axis=1)

                if debug_show_additional_tiepoints_subset:
                    dt.display_tiepoints(base_subset, other_subset, mkpts_sub, mconf_sub,
                                         title=f"new tiepoints for subset ({y_counter}/{x_counter})")

                mkpts_sub[:, 0] = mkpts_sub[:, 0] * 1 / sub_resize_factor_1
                mkpts_sub[:, 1] = mkpts_sub[:, 1] * 1 / sub_resize_factor_1
                mkpts_sub[:, 2] = mkpts_sub[:, 2] * 1 / sub_resize_factor_2
                mkpts_sub[:, 3] = mkpts_sub[:, 3] * 1 / sub_resize_factor_2
                mkpts = mkpts.astype(int)

                # adapt the tie points of the other image to account that we look at a subset
                mkpts_sub[:, 0] = mkpts_sub[:, 0] + min_x
                mkpts_sub[:, 1] = mkpts_sub[:, 1] + min_y
                mkpts_sub[:, 2] = mkpts_sub[:, 2] + res_min_x
                mkpts_sub[:, 3] = mkpts_sub[:, 3] + res_min_y

                # add the tie points and confidence values to the list
                additional_tiepoints.append(mkpts_sub)
                for elem in mconf_sub:
                    additional_confidences.append(elem)

                if verbose:
                    print(f"  {mkpts_sub.shape[0]} additional tie-points matched.")

        if (len(additional_tiepoints) != 0:
            # stack tie points together
            additional_tiepoints = np.vstack(additional_tiepoints)

            # stack all confidences
            additional_confidences = np.array(additional_confidences)

            # remove duplicates
            additional_tiepoints, unique_indices = np.unique(additional_tiepoints, return_index=True, axis=0)
            additional_tiepoints = additional_tiepoints.astype(int)
            additional_confidences = additional_confidences[unique_indices]

            additional_tiepoints, additional_confidences = filter_with_mask(additional_tiepoints, additional_confidences, mask_1, mask_2)

            # stack with all tiepoints
            all_tiepoints = np.concatenate((all_tiepoints, additional_tiepoints))
            all_confidences = np.concatenate((all_confidences, additional_confidences))

            # remove duplicates again
            all_tiepoints, unique_indices = np.unique(all_tiepoints, return_index=True, axis=0)
            all_tiepoints = all_tiepoints.astype(int)
            all_confidences = all_confidences[unique_indices]

        # apply threshold
        if min_threshold is not None:

            num_tp_before = all_tiepoints.shape[0]
            all_tiepoints = all_tiepoints[all_confidences >= min_threshold]
            all_confidences = all_confidences[all_confidences >= min_threshold]
            num_tp_after = all_tiepoints.shape[0]

            if verbose:
                print(f"{num_tp_before - num_tp_after} tiepoints removed with a quality lower than {min_threshold}.")

        # filter for outliers
        if filter_outliers and all_tiepoints.shape[0] >= 4:
            _, mask = cv2.findHomography(all_tiepoints[:, 0:2], all_tiepoints[:, 2:4], cv2.RANSAC, 5.0)
            mask = mask.flatten()

            if verbose:
                print(f"{np.count_nonzero(mask)} outliers removed.")

            # 1 means outlier
            all_tiepoints = all_tiepoints[mask == 0]
            all_confidences = all_confidences[mask == 0]

    if verbose:
        print(f"In total {all_tiepoints.shape[0]} tie-points were found "
              f"with a quality of {round(np.average(all_confidences), 3)}")

    if debug_show_final_tiepoints:
        dt.display_tiepoints(input_img1, input_img2, all_tiepoints, all_confidences,
                             title=f"Final tie points {all_tiepoints.shape[0]}")

    return all_tiepoints, all_confidences
