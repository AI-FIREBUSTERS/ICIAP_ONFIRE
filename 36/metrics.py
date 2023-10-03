import os

class Metrics:
    """
    A class used to calculate and store various performance metrics.
    """

    def __init__(self):
        """
        Initializes all metrics and constants.
        """
        self.tp, self.fp, self.fn = 0, 0, 0
        self.delays, self.predicted, self.truth = [], [], []
        self.p, self.r, self.d, self.dn = 0, 0, 0, 0
        self.pfr, self.pfr_delta = 0, 0
        self.mem_delta = 0
        self.fds = 0
        self.DELTA = 5
        self.PFR_TARGET = 10
        self.MEM_TARGET = 4

    def _computeTpFpTn(self):
        """
        Computes the true positives, false positives and false negatives.
        """
        for result in os.listdir("results/"):
            with open("results/"+result, "r") as f:
                row = f.read()
                if len(row) != 0:
                    self.predicted.append(int(row))
                else:
                    self.predicted.append(-1)

        for result in os.listdir("labels/"):
            with open("labels/"+result, "r") as f:
                row = f.read()
                if len(row) != 0:
                    self.truth.append(int(row))
                else:
                    self.truth.append(-1)

        for i in range(len(self.truth)):
            val_truth, val_predicted = self.truth[i], self.predicted[i]
            if val_truth != -1 and val_predicted != -1:
                if val_predicted >= max(0, val_truth-self.DELTA):
                    self._computeDelay(val_truth, val_predicted)
                    self.tp += 1
            elif (val_truth == -1 and val_predicted != -1) or (val_truth != -1 and val_predicted != -1 and val_predicted < max(0, val_truth-self.DELTA)):
                self.fp += 1
            elif val_truth != -1 and val_predicted == -1:
                self.fn += 1
    
    def _computeDelay(self, val_truth, val_predicted):
        """
        Computes and stores the delay between the predicted and actual value.
        """
        self.delays.append(abs(val_predicted-val_truth))

    def _computePrecision(self):
        """
        Computes the precision of the model.
        """
        self.p = self.tp/(self.tp+self.fp)
    
    def _computeRecall(self):
        """
        Computes the recall of the model.
        """
        self.r = self.tp/(self.tp+self.fn)

    def _computeNotificationDelay(self): 
        """
        Computes the average notification delay of the model.
        """
        self.d = sum(self.delays)/self.tp

    def _computeNormalizedNotificationDelay(self):
        """
        Computes the normalized average notification delay of the model.
        """
        self.dn = max(0, 60 - self.d)/60

    def computePfr(self, total_frames, processing_time_frames):
        """
        Computes the processing frame rate (PFR) of the model.
        """
        self.pfr = total_frames/processing_time_frames

    def _computePfrDelta(self):
        """
        Computes the delta processing frame rate of the model.
        """
        self.pfr_delta = max(0, (self.PFR_TARGET/self.pfr) - 1)

    def _computeMemDelta(self, mem = 1):
        """
        Computes the delta memory usage of the model.
        """
        self.mem_delta = max(0, (mem/self.MEM_TARGET) - 1)

    def _computeFds(self):
        """
        Computes the fire detection score (FDS) of the model.
        """
        self.fds = (self.p * self.r * self.dn)/((1 + self.pfr_delta) * (1 + self.mem_delta))

    def _compute_all(self):
        """
        Calls all the computation methods to calculate all metrics.
        """
        self._computeTpFpTn()
        self._computePrecision()
        self._computeRecall()
        self._computeNotificationDelay()
        self._computeNormalizedNotificationDelay()
        self._computePfrDelta()
        self._computeMemDelta()
        self._computeFds()

    def printResults(self):
        """
        Prints all the calculated metrics in a clean, organized manner.
        """
        self._compute_all()
        print("Performance Metrics:\n-------------------")
        print("True Positives (TP): {}".format(self.tp))
        print("False Positives (FP): {}".format(self.fp))
        print("False Negatives (FN): {}".format(self.fn))
        print("Precision (P): {:.2f}".format(self.p))
        print("Recall (R): {:.2f}".format(self.r))
        print("Avarage Notification Delay (D): {:.2f}".format(self.d))
        print("Normalized Avarage Notification Delay (DN): {:.2f}".format(self.dn))
        print("Processing Frame Rate (PFR): {:.2f}".format(self.pfr))
        print("Delta Processing Frame Rate (PFR_DELTA): {:.2f}".format(self.pfr_delta))
        print("Delta Memory (MEM_DELTA): {:.2f}".format(self.mem_delta))
        print("Fire Detection Score (FDS): {:.2f}".format(self.fds))


if __name__ == "__main__":
    metrics = Metrics()
    metrics.printResults()