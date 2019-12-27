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

import swiftwatcher.image_processing.data_structures as ds


class SegmentTracker:
    """A class which stores two segmented frames, and provides methods
    for matching the segments between two frames. When a new frame is
    added as "frame 1", the previous frame 1 becomes frame 2, allowing
    for tracking through longer sequences of frames.

    Functions that explicitly use frame segments are treated as class
    methods, while more generic functions are stored separately."""

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
        """Formulate a matrix containing costs for every combination
        of segments within two frames. Example: Matching 4 segments in
        the current frame to 3 segments in the previous frame.

                                  current
                             frame's segments
                                1  2  3  4    _
        previous   1  [D][ ][ ][!][!][!][!]    |
         frame's   2  [ ][D][ ][!][!][!][!]    |
        segments   3  [ ][ ][D][!][!][!][!]    |
                      [ ][ ][ ][A][ ][ ][ ]    |-- Cost Matrix
                      [ ][ ][ ][ ][A][ ][ ]    |
                      [ ][ ][ ][ ][ ][A][ ]    |
                      [ ][ ][ ][ ][ ][ ][A]   _|

        -[!]: Costs associated with two segments matching. (More likely
            matches have lower associated costs.)
        -[D]/[A]: Costs associated with two segments having no match.
            (Disappeared and appeared respectively.)
        -[ ]: Impossible spaces. (These are filled with values such that
            it is impossible for them to be chosen by the matching
            algorithm.)

        By using the Hungarian algorithm, this matrix is solved to give
        an ideal outcome (match, appear, disappear) for each segment in
        both frames. The total sum of selected costs will be the minimum
        across all possible combinations of matches, as per the
        Assignment Problem. An example set of matches could be:

            -Seg #1 (prev) -> Seg #1 (curr)         (! value selected),
                              Seg #2 (curr) appears (A value selected)
            -Seg #2 (prev) -> Seg #3 (curr)         (! value selected),
            -Seg #3 (prev) -> Seg #4 (curr)         (! value selected),
        """

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

    def store_assignments(self, assignments):
        """Take the output of "linear_sum_assignment" and convert it
        into human-readable labels (match, disappear, appear), then
        store those assignments in the segment objects themselves."""

        n_prev = self.get_cached_frame().get_num_segments()

        # Interpreting assignments is a bit hard to grasp, because
        # "assignments" are a mapping of row indexes to column indexes,
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
        """Transfer the segment history from a previous segment to a
        matched segment in a subsequent frame."""

        for (i, segment) in enumerate(self.current_frame.segments):
            # If it hasn't "A"ppeared, then it has a match
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
        """See if any segments that have no match (have disappeared
        from frame) meet the conditions to be considered an event:
            1. Segment must have disappeared within chimney ROI
            2. Segment must have been previously matched with another
            segment."""

        for segment in self.cached_frame.segments:
            if segment.status == "D":
                # Condition 1
                pos = segment.centroid
                if self.roi_mask[int(pos[0]), int(pos[1])] != 255:
                    continue

                # Condition 2
                if len(segment.segment_history) < 1:
                    continue

                # Both conditions met, so append segment to its own history and
                # store the motion path within the list of detected events
                event_motion_path = segment.segment_history
                event_motion_path.append(segment)
                self.detected_events.append(event_motion_path)


def intialize_cost_matrix(n_curr, n_prev):
    """Initialize a square cost matrix with size equal to the total
    segments across both frames. Values set to slightly larger than the
    default "no match" value."""

    n_total = n_curr + n_prev

    return np.ones((n_total, n_total)) + sys.float_info.epsilon


def calculate_distance_cost(segment_curr, segment_prev):
    """Map the distance between segments into a cost for the cost
    matrix. Higher distances mean larger costs."""

    dist = distance.euclidean(segment_prev.centroid,
                              segment_curr.centroid)
    dist_cost = 2 ** (dist - 25)

    return dist_cost


def calculate_angle_cost(segment_curr, segment_prev):
    """Compare the angle of the vector between seg_curr and seg_prev to
    the angle of vector associated with the existing motion path. Angle
    difference falls within range [0, 180]. Example:

                            *      \
                              *    |  Motion path vector
                             *     \
            Vector       |    o    V        . = current-frame segment
       between segments  V    .             o = previous-frame segment
                                            * = prior matched segments

    There is a low cost if the (o, .) vector is similar to the (*, o)
    vector. (i.e. <90 degrees)"""

    if len(segment_prev.segment_history) > 0:
        # Get the (x, y) coordinates of:
        #     -the current-frame segment
        #     -the previous-frame segment it's being compared to
        #     -the first segment in the chain of prior matched segments
        curr_pos = segment_curr.centroid
        prev_pos = segment_prev.centroid
        initial_pos = segment_prev.segment_history[0].centroid

        # Calculate the angle of the vector of the existing motion path
        del_y = initial_pos[0] - prev_pos[0]
        del_x = initial_pos[1] - prev_pos[1]
        old_angle = math.degrees(math.atan2(del_y, -1*del_x))

        # Calculate the angle of the vector connecting the compared segments
        del_y = prev_pos[0] - curr_pos[0]
        del_x = prev_pos[1] - curr_pos[1]
        new_angle = math.degrees(math.atan2(del_y, -1*del_x))

        angle_difference = abs(new_angle - old_angle)
        # The first calculation falls within range [0, 360]. But, a 360*
        # difference is equivalent to a 0* difference, so the second
        # calculation corrects this.
        angle_difference = min(angle_difference, 360 - angle_difference)

        # Map differences <90 to a low cost, and >90 to a high cost
        angle_cost = 2 ** (angle_difference - 90)

    else:
        # No prior matched segments, so use a default value
        angle_cost = 1

    return angle_cost


def calculate_nonmatch_cost():
    """Current costs for segment pairs not matching is set to a
    default value of 1."""

    return 1


def apply_hungarian_algorithm(cost_matrix):
    """Apply the Hungarian algorithm to find an optimal set of
    matches, as outlined in formulate_cost_matrix."""

    _, assignments = linear_sum_assignment(cost_matrix)

    return assignments
