
import cv2
from os.path import join
from pathlib import Path
import matplotlib.pyplot as plt
from beesypollen.detection import (
    STANDARDIZED_POLLEN_DIM_M11_06,
    STANDARDIZED_WIDTH_M11_06,
    STANDARDIZED_HEIGHT_M11_06,
    detect_pollen_m11_06,
)
from beesypollen.color import \
    extract_pollen_color, COLOR_EXTRACTION_CRIT, COLOR_EXTRACTION_METHOD
from beesypollen.clustering import \
    best_cluster, arbitrary_clusters, ClusterMode


def test():
    path = join("test_resources", "test_img.jpg")
    log_dir = "output"
    Path(log_dir).mkdir(exist_ok=True, parents=True)

    img = cv2.imread(path)
    img_raw_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (
        std_upper_left,
        std_lower_right,
        pollen_detections_xy,
        std_image_rgb
    ) = detect_pollen_m11_06(
        img_raw_rgb=img_raw_rgb,
        log_dir=log_dir,
    )

    df_pollen_colors = extract_pollen_color(
        img_raw_rgb=img_raw_rgb,
        pollen_detections_xy=pollen_detections_xy,
        primary_crit=COLOR_EXTRACTION_CRIT.SIZE,
        # method=COLOR_EXTRACTION_METHOD.GMM,  # three times slower
        method=COLOR_EXTRACTION_METHOD.KMEANS,  # three times faster
        pollen_dim=STANDARDIZED_POLLEN_DIM_M11_06,
        standardized_height=STANDARDIZED_HEIGHT_M11_06,
        standardized_width=STANDARDIZED_WIDTH_M11_06
    )

    # best_cluster_division contains the cluster labels
    linkage_matrix, best_cluster_division = best_cluster(
        df_pollen_colors,
        useCalibratedColors=False,
        cluster_by=ClusterMode.PRIMARY
    )

    # ignore, only useful for Pollenyzer app
    arbitrary_cluster_result, diversity = arbitrary_clusters(
        linkage_matrix=linkage_matrix,
        df_pollen_colors=df_pollen_colors,
        useCalibratedColors=False,
        n_clusters=(
            None
            if len(best_cluster_division) == 0
            else range(1, max(20, best_cluster_division.max() + 5))
        ),
    )

