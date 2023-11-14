import numpy as np

# Segini doang tubes nya :D, ga deng canda ampun dah


# Mengubah data RGB dari suatu image menjadi data histogram H, S, dan V.
# Menerima masukan berupa image dengan format array of RGB
# Menghasilkan keluaran berupa histogram H, S, dan V
def rgb_to_hsv_hist(rgb_image):
    # Normalize RGB values to the range [0, 1]
    normalized_image = rgb_image / 255.0

    # Find Cmax, Cmin, and ∆
    Cmax = np.max(normalized_image, axis=-1)
    Cmin = np.min(normalized_image, axis=-1)
    delta = Cmax - Cmin

    # Calculate Hue (H)
    H = np.zeros_like(Cmax)
    non_zero_delta = delta != 0

    H[non_zero_delta] = np.where(
        Cmax[non_zero_delta] == normalized_image[non_zero_delta, ..., 0],
        (
            (
                (
                    normalized_image[non_zero_delta, ..., 1]
                    - normalized_image[non_zero_delta, ..., 2]
                )
                / delta[non_zero_delta]
            )
            % 6.0
        ),
        H[non_zero_delta],
    )

    H[non_zero_delta] = np.where(
        Cmax[non_zero_delta] == normalized_image[non_zero_delta, ..., 1],
        (
            2.0
            + (
                normalized_image[non_zero_delta, ..., 2]
                - normalized_image[non_zero_delta, ..., 0]
            )
            / delta[non_zero_delta]
        ),
        H[non_zero_delta],
    )

    H[non_zero_delta] = np.where(
        Cmax[non_zero_delta] == normalized_image[non_zero_delta, ..., 2],
        (
            4.0
            + (
                normalized_image[non_zero_delta, ..., 0]
                - normalized_image[non_zero_delta, ..., 1]
            )
            / delta[non_zero_delta]
        ),
        H[non_zero_delta],
    )

    H = (H / 6.0) % 1.0

    # Calculate Saturation (S)
    S = np.zeros_like(Cmax)
    S[Cmax != 0] = delta[Cmax != 0] / Cmax[Cmax != 0]

    # Calculate Value (V)
    V = Cmax

    hist_h = np.histogram(
        H,
        bins=[
            0,
            25 / 360,
            40 / 360,
            120 / 360,
            190 / 360,
            270 / 360,
            295 / 360,
            315 / 360,
            360 / 360,
        ],
    )
    hist_s = np.histogram(S, bins=[0, 0.2, 0.7, 1])
    hist_v = np.histogram(V, bins=[0, 0.2, 0.7, 1])

    return hist_h, hist_s, hist_v