"""
    Contains functionality to track segments between frames, while also
    determining when a segment has appeared or disappeared within a
    frame.
"""

import sys
import math

import numpy as np
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

import swiftwatcher_refactor.image_processing.data_structures as ds


class SegmentTracker:
    def __init__(self, roi_mask):
        self.current_frame = None
        self.cached_frame = ds.Frame()  # Empty frame object

        # Used when detecting a "swift entered chimney" event
        self.roi_mask = roi_mask
        self.detected_events = []

    def get_current_frame(self):
        return self.current_frame

    def get_cached_frame(self):
        return self.cached_frame

    def set_current_frame(self, frame):
        self.current_frame = frame

    def cache_current_frame(self):
        self.cached_frame = self.current_frame

    def formulate_cost_matrix(self):
        current_frame = self.get_current_frame()
        previous_frame = self.get_cached_frame()
        n_curr = current_frame.get_num_segments()
        n_prev = previous_frame.get_num_segments()

        cost_matrix = intialize_cost_matrix(n_curr, n_prev)

        # Only calculate match costs if both frames have segments
        if n_curr > 0 and n_prev > 0:
            for i, segment_prev in enumerate(previous_frame.segments):
                for j, segment_curr in enumerate(current_frame.segments):
                    d_cost = calculate_distance_cost(segment_curr, segment_prev)
                    a_cost = calculate_angle_cost(segment_curr, segment_prev)

                    # The second index requires an offset, see matrix def
                    cost_matrix[i, j + n_prev] = 0.5*d_cost + 0.5*a_cost

        for i in range(n_curr + n_prev):
            cost_matrix[i, i] = calculate_nonmatch_cost()

        return cost_matrix

    def apply_hungarian_algorithm(self, cost_matrix):
        _, assignments = linear_sum_assignment(cost_matrix)

        return assignments

    def interpret_assignments(self, assignments):
        n_prev = self.get_cached_frame().get_num_segments()

        # Interpreting assignments is a bit hard to grasp, because
        # "assignments" is a mapping of row indexes to column indexes,
        # but those values aren't the same at the label of each segment.
        # (v - n_prev) accounts for the offset.
        prev_assignments = [(v - n_prev) for v in assignments[:n_prev]]
        curr_assignments = [(v - n_prev) for v in assignments[n_prev:]]

        for prev_label, assignment in enumerate(prev_assignments):
            # Previous-frame segment matched to valid current-frame segment
            if assignment >= 0:
                self.cached_frame.segments[prev_label].status = assignment
                self.current_frame.segments[assignment].status = prev_label

            # If not matched, it has disappeared
            else:
                self.cached_frame.segments[prev_label].status = "D"

        for curr_label, assignment in enumerate(curr_assignments):
            # Condition if current-frame segment has appeared
            if assignment == curr_label:
                self.current_frame.segments[curr_label].status = "A"

    def link_matching_segments(self):
        for (i, segment) in enumerate(self.current_frame.segments):
            if segment.status != "A":
                matched_segment = self.cached_frame.segments[segment.status]

                # This operation doesn't create a new object, so it appends
                # itself to its own segment_history list.
                # Done this way to prevent nested segment_history copies
                # which would be extremely memory inefficient.
                # This has the added benefit of retroactively updating the
                # segment histories of previous segments, too, because
                # each new segment in the chain uses the same history object.
                new_history = matched_segment.segment_history
                new_history.append(matched_segment)

                self.current_frame.segments[i].segment_history = new_history

    def check_for_events(self):
        for segment in self.cached_frame.segments:
            if segment.status == "D":
                if len(segment.segment_history) >= 2:
                    pos = segment.regionprops.centroid
                    if self.roi_mask[int(pos[0]), int(pos[1])] == 255:
                        self.detected_events.append(segment)


def intialize_cost_matrix(n_curr, n_prev):
    n_total = n_curr + n_prev

    return np.ones((n_total, n_total)) + sys.float_info.epsilon


def calculate_distance_cost(segment_curr, segment_prev):
    dist = distance.euclidean(segment_prev.regionprops.centroid,
                              segment_curr.regionprops.centroid)
    dist_cost = 2 ** (dist - 25)

    return dist_cost


def calculate_angle_cost(segment_curr, segment_prev):
    """Compare the angle of the vector between seg_curr and seg_prev to
    the angle of vector associated with the existing motion path. Angle
    difference falls within range [0, 180]. Example:

                            *      \
                              *    |  Motion path vector
                             *     \
            Vector       |    o    V
       between segments  V    .

    Where * is a previously matched segment, o is the previous segment,
    and . is the current segment. There is a low cost if the (o, .)
    vector is similar to the (*, o) vector. (< 90 degrees)"""

    if len(segment_prev.segment_history) > 0:
        # Get the (x, y) coordinates of:
        #     -the current-frame segment
        #     -the previous-frame segment it's being compared to
        #     -the initial segment in the chain of matched segments
        curr_pos = segment_curr.regionprops.centroid
        prev_pos = segment_prev.regionprops.centroid
        initial_pos = segment_prev.segment_history[-1].regionprops.centroid

        # Calculate the angle of the vector of the existing motion path
        del_y = prev_pos[0] - initial_pos[0]
        del_x = prev_pos[1] - initial_pos[1]
        old_angle = math.degrees(math.atan2(del_y, -1*del_x))

        # Calculate the angle of the vector connecting the compared segments
        del_y = curr_pos[0] - prev_pos[0]
        del_x = curr_pos[1] - prev_pos[1]
        new_angle = math.degrees(math.atan2(del_y, -1*del_x))

        angle_difference = abs(new_angle - old_angle)
        # The first calculation falls within range [0, 360]. But, a 360*
        # difference is equivalent to a 0* difference, so the second
        # calculation corrects this.
        angle_difference = min(angle_difference, 360 - angle_difference)

        # Map differences <90 to a low cost, and >90 to a high cost
        angle_cost = 2 ** (angle_difference - 90)

    else:
        # Motion path vector doesn't exist, so use a default value
        angle_cost = 1

    return angle_cost


def calculate_nonmatch_cost():
    return 1
