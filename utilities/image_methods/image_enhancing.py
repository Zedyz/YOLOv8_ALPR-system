import cv2


class ImageEnhancing:
    def __init__(self):
        pass

    def upscale_gaussian_pyramid(self, image, n):
        upscaled_image = image
        for _ in range(n):
            upscaled_image = cv2.pyrUp(upscaled_image)
        return upscaled_image

    def increase_contrast(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(45, 45))
        enhanced_gray = clahe.apply(gray)
        _, binary_image = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary_image

    def apply_adaptive_thresholding(self, image):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        adaptive_thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, 11, 2)

        return adaptive_thresh_image
