import numpy as np
import os
import matplotlib.pyplot as plt

class ResultEvaluator():
    """
    ResultEvaluator class is used to evaluate the results of fire/smoke detection from frames of a video.
    """
    def __init__(self, verbose=True, confidence_threshold=0.3):
        """
        Initialize the instance variables.

        Args:
            verbose (bool): If True, prints additional log information.
            fire_threshold (int): Number of consecutive frames before declaring a fire.
            confidence_threshold (float): Probability threshold for the detection.
        """
        self.fire_counter = 0
        self.fire_smoke_detected = False
        self.smoke_frame_counter = 0
        self.fire_frame_counter = 0
        self.fire_threshold = 0

        self.verbose = verbose
        self.confidence_threshold = confidence_threshold

    @property
    def detected(self):
        """
        Property that returns the fire/smoke detection status.
        """
        return self.fire_smoke_detected

    def set_frame_dimensions(self, width, height):
        """
        Sets the dimensions of the frame.

        Args:
            width (int): Width of the frame.
            height (int): Height of the frame.
        """
        self.width = width
        self.height = height

    def reset(self):
        """
        Resets the counters and accumulation matrices.
        """
        self.fire_smoke_detected = False
        self.fire_counter = 0
        self.smoke_frame_counter = 0
        self.fire_frame_counter = 0
        self.fire_accumulation_matrix = np.zeros((self.width,self.height))
        self.smoke_accumulation_matrix = np.zeros((self.width,self.height))
        self.fire_weight_matrix = np.zeros((self.width,self.height))
        self.smoke_weight_matrix = np.zeros((self.width,self.height))

    def evaluate(self, result, frame_counter, video_name = None):
        """
        Evaluate the given result and increment the counters and accumulation matrices if fire or smoke is detected.

        Args:
            result: Object with detection result information.
            frame_counter: Current frame counter.
            video_name: Name of the video file being processed.
        """
        # Extract result information
        info = result.pandas().xyxy[0].to_dict(orient = "records")

        if len(info) != 0:
            # Increment the fire counter if fire is detected
            self.fire_counter += 1
            for detected_element in info:
                self._accumulate(detected_element, video_name, frame_counter)

                if self.verbose:
                    print(f"Detected element: {detected_element['name']} with confidence: {detected_element['confidence']} in frame: {frame_counter}")
        else:
            self.fire_counter = 0
            self.smoke_frame_counter = 0
            self.fire_frame_counter = 0
            self.fire_accumulation_matrix = np.zeros((self.width,self.height))
            self.smoke_accumulation_matrix = np.zeros((self.width,self.height))
            self.fire_weight_matrix = np.zeros((self.width,self.height))
            self.smoke_weight_matrix = np.zeros((self.width,self.height))  

        # Check if fire or smoke detection meets the threshold
        if self.fire_counter >= self.fire_threshold:
            self.fire_detected = self._analyze_accumulation_matrix(self.fire_accumulation_matrix, self.fire_weight_matrix) if self.fire_frame_counter > 0 else False
            self.smoke_detected = self._analyze_accumulation_matrix(self.smoke_accumulation_matrix, self.smoke_weight_matrix) if self.smoke_frame_counter > 0 else False

            if self.verbose:
                print(f"Fire detected: {self.fire_detected}")
                print(f"Smoke detected: {self.smoke_detected}")

            if self.fire_detected or self.smoke_detected:
                self.fire_smoke_detected = True
    
    def _analyze_accumulation_matrix(self, accumulation_matrix, weight_matrix):
        """
        Analyzes the accumulation matrix and decides if there is a fire or smoke.

        Args:
            accumulation_matrix: The accumulation matrix to be analyzed.
            weight_matrix: The weight matrix associated with the accumulation matrix.

        Returns:
            bool: True if fire/smoke detected, False otherwise.
        """
        # Compute the weighted average
        weighted_average_matrix = np.divide(
            accumulation_matrix, 
            weight_matrix, 
            out=np.zeros_like(accumulation_matrix), 
            where=weight_matrix!=0
        )

        return np.max(weighted_average_matrix) >= self.confidence_threshold

    def _accumulate(self, detected_element, video_name=None, frame_counter=None):
        """
        Adds the confidence of the prediction to the corresponding pixel in the accumulation matrix.

        Args:
            detected_element: Detected element information.
            video_name: Name of the video file being processed.
            frame_counter: The current frame counter.
        """

        # Compute the weight based on the inverse of the confidence
        confidence = detected_element["confidence"]
        weight = 1.0 / confidence

        if detected_element["name"] == "fire":

            self.fire_accumulation_matrix[round(detected_element["ymin"]):round(detected_element["ymax"]), 
                                      round(detected_element["xmin"]):round(detected_element["xmax"])] += detected_element["confidence"] * weight
            self.fire_weight_matrix[round(detected_element["ymin"]):round(detected_element["ymax"]), 
                                round(detected_element["xmin"]):round(detected_element["xmax"])] += weight
            self.fire_frame_counter += 1

            if video_name is not None:
                if not os.path.exists(f"fire_matrices/{video_name}"):
                    os.makedirs(f"fire_matrices/{video_name}")

                self.print_accumulation_matrix(self.fire_accumulation_matrix, f"fire_matrices/{video_name}/{frame_counter}.png")

        elif detected_element["name"] == "smoke":

            self.smoke_accumulation_matrix[round(detected_element["ymin"]):round(detected_element["ymax"]), 
                                       round(detected_element["xmin"]):round(detected_element["xmax"])] += detected_element["confidence"] * weight
            self.smoke_weight_matrix[round(detected_element["ymin"]):round(detected_element["ymax"]), 
                                    round(detected_element["xmin"]):round(detected_element["xmax"])] += weight
            self.smoke_frame_counter += 1

            if video_name is not None:
                if not os.path.exists(f"smoke_matrices/{video_name}"):
                    os.makedirs(f"smoke_matrices/{video_name}")

                self.print_accumulation_matrix(self.smoke_accumulation_matrix, f"smoke_matrices/{video_name}/{frame_counter}.png")

    def print_accumulation_matrix(self, matrix, file_path):
        """
        Save the given accumulation matrix as an image.

        Args:
            matrix: Accumulation matrix to be saved.
            file_path: File path to save the image.
        """
        plt.imshow(matrix, origin='upper', cmap='gray')
        plt.axis('on')
        plt.savefig(file_path, bbox_inches='tight')

    def set_fire_threshold(self, fire_threshold):
        """
        Sets the fire threshold.

        Args:
            fire_threshold: Number of consecutive frames before declaring a fire.
        """
        self.fire_threshold = fire_threshold