import numpy as np


def process_pose_data(joint2d_seq_list, confidence_thresh=0.2, vid_paths=None):
    for i, op_seq in enumerate(joint2d_seq_list):
        cur_joint2d_seq = joint2d_seq_list[i][:, :, :2]  # don't filter confidence
        cur_confidence_seq = joint2d_seq_list[i][:, :, 2]

        # fix missing (very low confidence) frames caused by occlusion
        # to do this linearly interpolate from surrounding high-confidence frames
        num_frames = cur_joint2d_seq.shape[0]
        for joint_idx in range(cur_joint2d_seq.shape[1]):
            t = 0
            while t < num_frames:
                if cur_confidence_seq[t, joint_idx] < confidence_thresh:
                    # find the next valid frame
                    next_valid_frame = t + 1
                    while (
                        next_valid_frame < num_frames
                        and cur_confidence_seq[next_valid_frame, joint_idx]
                        < confidence_thresh
                    ):
                        next_valid_frame += 1
                    # then update sequence of invalid frames accordingly
                    init_valid_frame = t - 1
                    if t == 0 and next_valid_frame == num_frames:
                        # worst case scenario, every frame is bad
                        # can't do anything
                        pass
                    elif t == 0:
                        # bad frames to start the sequence
                        # set all t < next_valid_frame to next_valid_frame
                        cur_joint2d_seq[
                            :next_valid_frame, joint_idx, :
                        ] = cur_joint2d_seq[next_valid_frame, joint_idx, :].reshape(
                            (1, 2)
                        )
                    elif next_valid_frame == num_frames:
                        # bad until the end
                        # set all t > init_valid_frame to init_valid_frame
                        cur_joint2d_seq[
                            init_valid_frame:, joint_idx, :
                        ] = cur_joint2d_seq[init_valid_frame, joint_idx, :].reshape(
                            (1, 2)
                        )
                    else:
                        # have a section of >= 1 frame that's bad
                        # linearly interpolate in this section
                        step_size = 1.0 / (next_valid_frame - init_valid_frame)
                        cur_step = step_size
                        cur_t = t
                        while cur_t < next_valid_frame:
                            cur_joint2d_seq[cur_t, joint_idx, :] = (
                                1.0 - cur_step
                            ) * cur_joint2d_seq[
                                init_valid_frame, joint_idx, :
                            ] + cur_step * cur_joint2d_seq[
                                next_valid_frame, joint_idx, :
                            ]
                            cur_t += 1
                            cur_step += step_size

                    t = next_valid_frame
                else:
                    t += 1

        # then filter to smooth it out a bit
        # can additionally smooth if desired
        # cur_joint2d_seq = filter_poses(cur_joint2d_seq, fcmin=0.05, beta=0.005, freq=30)

        # update
        joint2d_seq_list[i][:, :, :2] = cur_joint2d_seq
        if i % 10 == 0:
            print(
                "Finished " + str(i + 1) + " of " + str(len(joint2d_seq_list)) + "..."
            )
    return joint2d_seq_list
