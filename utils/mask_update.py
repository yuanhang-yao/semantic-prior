import os
import shutil

import cv2
import numpy as np
import torch


def center_point_inside_contour(center_point, target_mask):
    center_y, center_x = center_point
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    center_index_XmergeY = {center_y * 1.0 + center_x * 0.0001}
    temp_contour_mask = np.zeros(target_mask.shape, np.uint8)

    overlap_found = False
    for target_contour in target_contours:
        target_contour_mask = np.zeros(target_mask.shape, np.uint8)
        cv2.fillPoly(target_contour_mask, [target_contour], (255))
        target_index = np.where(target_contour_mask == 255)
        target_index_XmergeY = set(target_index[0] * 1.0 + target_index[1] * 0.0001)
        if not center_index_XmergeY.isdisjoint(target_index_XmergeY):
            if cv2.contourArea(target_contour) > 50:
                break
            overlap_found = True
            cv2.fillPoly(temp_contour_mask, [target_contour], (255))
            break

    return temp_contour_mask, overlap_found


def process_image(y_and_x, y1_and_y2_and_x1_and_x2, img_shape, image, low_threshold=50, high_threshold=150, kernel_size=(3, 3), sigma=0):
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    edges = cv2.Canny(cv2.subtract(image, blurred_image), low_threshold, high_threshold)
    kernel = np.ones((3, 3), np.uint8)
    sparse_edges = cv2.dilate(edges, kernel, iterations=1)
    sparse_edges = cv2.erode(sparse_edges, kernel, iterations=1)

    temp_contour_mask_2 = np.zeros(img_shape, np.uint8)
    y1, y2, x1, x2 = y1_and_y2_and_x1_and_x2
    temp_contour_mask_2[y1:y2, x1:x2] = sparse_edges
    refine_mask, flag = center_point_inside_contour(y_and_x, temp_contour_mask_2)
    return refine_mask[y1:y2, x1:x2], flag


def initialize_point_labels(
    image_dir,
    point_dir,
    selected_image_dir,
    selected_mask_dir,
    selected_point_dir,
    unselected_image_dir=None,
    unselected_mask_dir=None,
    unselected_point_dir=None,
    crop_size=10,
    select_all=False,
    sample_names=None,
):
    for path in (selected_image_dir, selected_mask_dir, selected_point_dir):
        os.makedirs(path, exist_ok=True)
    if not select_all:
        if unselected_image_dir is None or unselected_mask_dir is None or unselected_point_dir is None:
            raise ValueError("Unselected output directories are required when select_all=False.")
        for path in (unselected_image_dir, unselected_mask_dir, unselected_point_dir):
            os.makedirs(path, exist_ok=True)

    if sample_names is None:
        input_img_list = np.sort(os.listdir(image_dir)).tolist()
    else:
        input_img_list = list(sample_names)

    for name in input_img_list:
        img = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(point_dir, name), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise RuntimeError(f"Failed to read image or mask: {name}")

        points = np.where(mask == 255)
        if len(points[0]) == 0:
            cv2.imwrite(os.path.join(selected_image_dir, name), img)
            cv2.imwrite(os.path.join(selected_mask_dir, name), mask)
            cv2.imwrite(os.path.join(selected_point_dir, name), mask)
            continue

        merged_result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        correct_point = 0

        for i in range(len(points[0])):
            center_y, center_x = points[0][i], points[1][i]
            x1, y1 = max(center_x - crop_size, 0), max(center_y - crop_size, 0)
            x2, y2 = min(center_x + crop_size, img.shape[1]), min(center_y + crop_size, img.shape[0])
            processed_roi, flag = process_image(
                (center_y, center_x),
                (y1, y2, x1, x2),
                mask.shape,
                img[y1:y2, x1:x2],
                low_threshold=20,
                high_threshold=40,
                kernel_size=(3, 3),
                sigma=0,
            )
            merged_result[y1:y2, x1:x2] = merged_result[y1:y2, x1:x2] + processed_roi
            if flag:
                correct_point = correct_point + 1

        if (correct_point / len(points[0])) >= (-1.0 if select_all else 0.8):
            merged_result = merged_result + mask
            merged_result = np.where(merged_result > 0, 255, 0).astype(np.uint8)
            cv2.imwrite(os.path.join(selected_image_dir, name), img)
            cv2.imwrite(os.path.join(selected_mask_dir, name), merged_result)
            cv2.imwrite(os.path.join(selected_point_dir, name), mask)
        else:
            cv2.imwrite(os.path.join(unselected_image_dir, name), img)
            cv2.imwrite(os.path.join(unselected_point_dir, name), mask)

    print("Initial data generated. Number of selected samples:", len(os.listdir(selected_image_dir)))


def update_gt_update_degen_corr(pred, gt_masks, thresh_Tb, thresh_k, size,degen=0.9):

    update_gt_masks = gt_masks.copy()

    num_labels, label_image = cv2.connectedComponents((gt_masks > 0.5).astype(np.uint8))

    background_kernel = np.ones((33, 33), np.uint8)
    target_kernel = np.ones((3, 3), np.uint8)
    max_limitation = size[0] * size[1] * 0.0015

    combined_thresh_mask = np.zeros_like(pred, dtype=np.float32)

    for region_num in range(1, num_labels):
        region_coords = np.argwhere(label_image == region_num)
        centroid = np.mean(region_coords, axis=0).astype(int)

        cur_point_mask = np.zeros_like(pred, dtype=np.uint8)
        cur_point_mask[centroid[0], centroid[1]] = 1

        nbr_mask = cv2.dilate(cur_point_mask, background_kernel) > 0
        targets_mask = cv2.dilate(cur_point_mask, target_kernel) > 0

        region_size_ratio = len(region_coords) / max_limitation
        threshold_start = (pred * nbr_mask).max() * thresh_Tb
        threshold_delta = thresh_k * ((pred * nbr_mask).max() - threshold_start) * region_size_ratio
        threshold = threshold_start + threshold_delta
        threshold = threshold.cpu().numpy() if isinstance(threshold, torch.Tensor) else threshold

        thresh_mask = (pred * nbr_mask > threshold).astype(np.float32)

        num_labels_thresh, label_image_thresh = cv2.connectedComponents(thresh_mask.astype(np.uint8))
        for num_cur in range(1, num_labels_thresh):
            curr_mask = (label_image_thresh == num_cur).astype(np.float32)
            if np.sum(curr_mask * targets_mask) == 0:
                thresh_mask -= curr_mask

        combined_thresh_mask = np.maximum(combined_thresh_mask, thresh_mask)

    target_patch = (update_gt_masks * combined_thresh_mask + pred * combined_thresh_mask) / 2
    background_patch = update_gt_masks * (1 - combined_thresh_mask)* degen
    update_gt_masks = background_patch + target_patch

    update_gt_masks = np.maximum(update_gt_masks, (gt_masks == 1).astype(np.float32))

    return update_gt_masks


def contour_index_sets(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    index_sets = []
    for contour in contours:
        contour_mask = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(contour_mask, [contour], (255))
        index = np.where(contour_mask == 255)
        index_sets.append(set(index[0] * 1.0 + index[1] * 0.0001))
    return contours, index_sets


def predicted_mask_matches_points(copy_mask, target_mask, lose_point_ratio=0.2, alarm_point_ratio=0.2):
    copy_contours, copy_index_sets = contour_index_sets(copy_mask)
    target_contours, target_index_sets = contour_index_sets(target_mask)
    if len(copy_contours) == 0:
        return True

    overwrite_contours = []
    un_overwrite_contours = []
    for contour, copy_index_set in zip(copy_contours, copy_index_sets):
        if any(not copy_index_set.isdisjoint(target_index_set) for target_index_set in target_index_sets):
            overwrite_contours.append(contour)
        else:
            un_overwrite_contours.append(contour)

    if len(un_overwrite_contours) / len(copy_contours) > lose_point_ratio:
        return False
    if (len(target_contours) - len(overwrite_contours)) / len(copy_contours) > alarm_point_ratio:
        return False
    return True


def select_hard_samples(
    unselected_image_dir,
    unselected_mask_dir,
    unselected_point_dir,
    lose_point_ratio=0.2,
    alarm_point_ratio=0.2,
):
    selected_names = []
    for name in np.sort(os.listdir(unselected_image_dir)):
        mask = cv2.imread(os.path.join(unselected_point_dir, name), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(unselected_mask_dir, name), cv2.IMREAD_GRAYSCALE)
        if mask is None or pred is None:
            continue
        points = np.where(mask == 255)
        if len(points[0]) == 0 or predicted_mask_matches_points(mask, pred, lose_point_ratio, alarm_point_ratio):
            selected_names.append(name)
    return selected_names


def merge_prediction_with_points(copy_mask, target_mask):
    copy_contours, copy_index_sets = contour_index_sets(copy_mask)
    _, target_index_sets = contour_index_sets(target_mask)
    copy_contour_mask_out = np.zeros(copy_mask.shape, np.uint8)
    for contour, copy_index_set in zip(copy_contours, copy_index_sets):
        if any(not copy_index_set.isdisjoint(target_index_set) for target_index_set in target_index_sets):
            cv2.fillPoly(copy_contour_mask_out, [contour], (255))
    copy_contour_mask_out = copy_contour_mask_out + target_mask
    return np.where(copy_contour_mask_out > 0, 255, 0).astype(np.uint8)


def refine_generated_masks(unselected_mask_dir, unselected_point_dir, selected_names):
    for name in selected_names:
        pred_mask_path = os.path.join(unselected_mask_dir, name)
        points_path = os.path.join(unselected_point_dir, name)
        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        points = cv2.imread(points_path, cv2.IMREAD_GRAYSCALE)
        if pred_mask is None or points is None:
            continue
        cv2.imwrite(pred_mask_path, merge_prediction_with_points(pred_mask, points))
    print("Generated labels refined.")


def move_hard_samples(
    unselected_image_dir,
    unselected_mask_dir,
    unselected_point_dir,
    selected_image_dir,
    selected_mask_dir,
    selected_point_dir,
    selected_names,
):
    for src_dir, dst_dir in (
        (unselected_image_dir, selected_image_dir),
        (unselected_mask_dir, selected_mask_dir),
        (unselected_point_dir, selected_point_dir),
    ):
        for name in selected_names:
            src = os.path.join(src_dir, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dst_dir, name))
                os.remove(src)
