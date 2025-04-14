from scipy.optimize import linear_sum_assignment
import numpy as np

class SpeakerMatcher:
    """
    Matches hypothesized segments (from pyannote diarization) with ground truth
    segments to assign the correct speaker identifier.
    """
    def __init__(self, hyp_segments, gt_segments):
        """
        Args:
            hyp_segments (list of dict): [{'start': float, 'end': float, 'speaker': str}, ...]
            gt_segments (list of dict): [{'start': float, 'end': float, 'speaker_id': str}, ...]
        """
        # Copy segments to avoid side effects.
        self.hyp_segments = [dict(seg) for seg in hyp_segments]
        self.gt_segments = [dict(seg) for seg in gt_segments]
        self.overlap_matrix = None
        self.matches = None

    @staticmethod
    def _compute_overlap(hyp_start, hyp_end, gt_start, gt_end):
        start = max(hyp_start, gt_start)
        end = min(hyp_end, gt_end)
        return max(0.0, end - start)

    def compute_overlap_matrix(self):
        num_hyp = len(self.hyp_segments)
        num_gt = len(self.gt_segments)
        self.overlap_matrix = np.zeros((num_hyp, num_gt))
        for i, hyp in enumerate(self.hyp_segments):
            for j, gt in enumerate(self.gt_segments):
                overlap = self._compute_overlap(hyp['start'], hyp['end'], gt['start'], gt['end'])
                # Normalize by the duration of the ground truth segment.
                duration = gt['end'] - gt['start']
                self.overlap_matrix[i, j] = overlap / duration if duration > 0 else 0.0

    def find_optimal_assignment(self, overlap_threshold=0.5):
        if self.overlap_matrix is None:
            self.compute_overlap_matrix()
        # Use Hungarian algorithm (maximize overlap -> minimize negative overlap)
        cost_matrix = -self.overlap_matrix
        # # save the self.overlap_matrix for debugging
        # np.save('overlap_matrix.npy', self.overlap_matrix)
        # print(asshole)
        hyp_indices, gt_indices = linear_sum_assignment(cost_matrix)
        self.matches = []
        for h, g in zip(hyp_indices, gt_indices):
            if self.overlap_matrix[h, g] >= overlap_threshold:  # Only accept matches above threshold
                self.matches.append((h, g))

    def assign_speaker_ids(self):
        if self.matches is None:
            self.find_optimal_assignment()
        # Initialize with None
        for seg in self.hyp_segments:
            seg['speaker_id'] = None
        for h, g in self.matches:
            self.hyp_segments[h]['speaker_id'] = self.gt_segments[g]['speaker_id']

    def get_labeled_segments(self):
        return self.hyp_segments