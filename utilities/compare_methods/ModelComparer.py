class ModelComparer:
    ground_truths = []
    matches = []

    def __init__(self, ground_truths_file_path):
        with open(ground_truths_file_path, 'r') as f:
            self.ground_truths = f.readlines()

    def compare(self, detected_texts_file_path):
        match = {
            "ground_truth": "",
            "match": "",
            "line_number": -1
        }

        detected_strings = []

        with open(detected_texts_file_path, "r") as f:
            detected_strings = f.readlines()

        for i, gt_line in enumerate(self.ground_truths):
            for j, detected_text in enumerate(detected_strings):
                if gt_line in detected_text:
                    match["ground_truth"] = gt_line
                    match["match"] = detected_text
                    match["line_number"] = j + 1
                    self.matches.append(match)
                    break

    def compare_to_gt(self):
        return len(self.matches) / len(self.ground_truths)