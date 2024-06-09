import os.path

import cv2


class Utils:
    def crop_image_license(self, image, obb_tensor, border_width=3):
        # Ensure the tensor is on the CPU before converting to NumPy array
        obb_coords = obb_tensor.cpu().numpy().reshape(-1, 2)

        xmin = int(obb_coords[:, 0].min()) + border_width
        xmax = int(obb_coords[:, 0].max()) - border_width
        ymin = int(obb_coords[:, 1].min()) + border_width
        ymax = int(obb_coords[:, 1].max()) - border_width

        xmin = max(xmin, 0)
        xmax = min(xmax, image.shape[1])
        ymin = max(ymin, 0)
        ymax = min(ymax, image.shape[0])

        cropped_image = image[ymin:ymax, xmin:xmax]

        return cropped_image

    def detect_characters(self, model, image_path, conf, spacing_threshold):
        results = model(image_path)

        value_to_char = [str(i) if i < 10 else chr(i + 55) for i in range(36)]

        # collect all detections, filtering those with conf > 0.5, and their x_min and x_max values
        detections_with_x_values = [(k, k.xyxy[0][0].item(), k.xyxy[0][2].item()) for r in results for k in r.obb if
                                    k.conf.item() > conf]

        # sort the detections based on the x_min values
        sorted_detections = sorted(detections_with_x_values, key=lambda x: x[1])

        # initialize an empty list to hold the detected characters
        detected_characters = []

        for i, (det, x_min, x_max) in enumerate(sorted_detections):
            char = value_to_char[int(det.cls.item())]
            detected_characters.append(char)

            if i < len(sorted_detections) - 1:
                next_x_min = sorted_detections[i + 1][1]
                space = next_x_min - x_max
                # check if the space indicates a missing character
                if space > spacing_threshold:
                    # estimate the number of missing characters based on the spacing
                    num_missing_chars = int(space / spacing_threshold)
                    # append placeholders for missing characters
                    detected_characters.extend("_" * num_missing_chars)

        return detected_characters

    def write_text_to_file(self, text, model, filename):
        """
        Writes an array of strings to a text file
        :param text: the array of strings
        :param filename: the name of the file
        """
        file_path = os.path.join(os.getcwd(), "experiments", "runs", model, filename)

        mode = "x" if not os.path.exists(file_path) else "a"

        print(f"Trying to {'create' if mode == 'x' else f'append to file at {file_path}...'}")

        try:
            with open(file_path, mode) as file:
                file.writelines(f'{line}\n' for line in text)
        except IOError as e:
            print(f"Failed to write to file at {file_path}. Eror: {str(e)}")

    def read_text_from_file(self, filename, model):
        pass


