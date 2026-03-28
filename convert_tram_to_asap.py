"""
Convert TRAM output (hps_track_0.npy) to ASAP-compatible AMASS .npz format.

TRAM gives us:
  - pred_rotmat: (N, 24, 3, 3) rotation matrices
  - pred_shape:  (N, 10) body shape per frame
  - pred_trans:  (N, 1, 3) translation

ASAP expects:
  - poses:           (N, 72) axis-angle (24 joints x 3)
  - betas:           (16,)   single averaged body shape
  - trans:           (N, 3)  translation
  - gender:          string
  - mocap_framerate: int

Usage:
  python convert_tram_to_asap.py <tram_result_name> [--fps 30]
  
Example:
  python convert_tram_to_asap.py IMG_7483
  python convert_tram_to_asap.py Eric_dab --fps 60
"""
import numpy as np
import sys
import os
import argparse

def rotmat_to_axis_angle(rotmat):
    """
    Convert rotation matrix to axis-angle representation.
    rotmat: (..., 3, 3) -> (..., 3)
    
    Think of it like this: a rotation matrix is a 3x3 grid of numbers
    that describes how to rotate something. Axis-angle is simpler:
    just 3 numbers that encode both WHICH axis to spin around and
    HOW MUCH to spin.
    """
    shape = rotmat.shape[:-2]
    rotmat = rotmat.reshape(-1, 3, 3)
    
    batch = rotmat.shape[0]
    axis_angle = np.zeros((batch, 3), dtype=np.float32)
    
    for i in range(batch):
        R = rotmat[i]
        # Angle from the trace of the rotation matrix
        cos_angle = (np.trace(R) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        if abs(angle) < 1e-6:
            # Nearly no rotation
            axis_angle[i] = np.zeros(3)
        elif abs(angle - np.pi) < 1e-6:
            # 180 degree rotation - special case
            # Find axis from the column of (R + I) with largest norm
            RpI = R + np.eye(3)
            col_norms = np.linalg.norm(RpI, axis=0)
            best_col = np.argmax(col_norms)
            axis = RpI[:, best_col]
            axis = axis / np.linalg.norm(axis)
            axis_angle[i] = axis * angle
        else:
            # General case: axis from skew-symmetric part
            axis = np.array([
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1]
            ])
            axis = axis / (2.0 * np.sin(angle))
            axis_angle[i] = axis * angle
    
    return axis_angle.reshape(*shape, 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='TRAM result folder name (e.g. IMG_7483)')
    parser.add_argument('--fps', type=int, default=30, help='Video framerate (default: 30)')
    parser.add_argument('--gender', default='neutral', help='Gender: male/female/neutral (default: neutral)')
    parser.add_argument('--smooth', type=int, default=5, help='Smoothing window size (0=off, default: 5)')
    args = parser.parse_args()
    
    # Load TRAM output
    tram_path = os.path.expanduser(f'~/Project/humanoid/tram/results/{args.name}/hps/hps_track_0.npy')
    if not os.path.exists(tram_path):
        print(f"Error: {tram_path} not found")
        sys.exit(1)
    
    print(f"Loading: {tram_path}")
    data = np.load(tram_path, allow_pickle=True).item()
    
    pred_rotmat = data['pred_rotmat']   # (N, 24, 3, 3)
    pred_shape = data['pred_shape']     # (N, 10)
    pred_trans = data['pred_trans']     # (N, 1, 3)
    
    # Convert tensors to numpy if needed
    if hasattr(pred_rotmat, 'numpy'):
        pred_rotmat = pred_rotmat.numpy()
        pred_shape = pred_shape.numpy()
        pred_trans = pred_trans.numpy()
    
    num_frames = pred_rotmat.shape[0]
    print(f"  {num_frames} frames")
    
    # --- Coordinate system fix ---
    # TRAM uses camera coords: X-right, Y-down, Z-forward (depth)
    # ASAP/AMASS uses world coords: X-right, Y-forward, Z-up
    # Correction: rotate 90 degrees around X axis
    # This is the same fix we used for Blender visualization
    R_correction = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ], dtype=np.float32)
    
    # Fix root orientation: pre-multiply root rotation by correction
    # Only joint 0 (root/pelvis) needs this -- other joints are relative to parent
    print("Fixing coordinate system (camera -> world)...")
    for i in range(num_frames):
        pred_rotmat[i, 0] = R_correction @ pred_rotmat[i, 0]
    
    # Fix translation: apply same rotation
    pred_trans = pred_trans.squeeze(1)  # (N, 3)
    trans = (R_correction @ pred_trans.T).T.astype(np.float32)  # (N, 3)
    
    # Zero out X and Y to prevent drift, keep Z (height)
    trans[:, 0] = 0.0
    trans[:, 1] = 0.0
    
    # 1. Rotation matrices -> axis-angle: (N, 24, 3, 3) -> (N, 72)
    print("Converting rotation matrices to axis-angle...")
    poses = rotmat_to_axis_angle(pred_rotmat)  # (N, 24, 3)
    poses = poses.reshape(num_frames, 72).astype(np.float32)
    
    # 2. Temporal smoothing to reduce TRAM noise
    if args.smooth > 1:
        print(f"Smoothing with window size {args.smooth}...")
        from scipy.ndimage import uniform_filter1d
        # Smooth poses (skip root orient indices 0-2 to avoid messing up orientation)
        poses = uniform_filter1d(poses, size=args.smooth, axis=0).astype(np.float32)
        # Smooth translation
        trans = uniform_filter1d(trans, size=args.smooth, axis=0).astype(np.float32)
    
    # 3. Average shape across frames, zero-pad from 10 to 16
    betas_avg = pred_shape.mean(axis=0)  # (10,)
    betas = np.zeros(16, dtype=np.float64)
    betas[:10] = betas_avg
    
    # Save
    out_dir = os.path.expanduser('~/Project/humanoid/ASAP/humanoidverse/data/motions/raw_tairantestbed_smpl/')
    out_path = os.path.join(out_dir, f'{args.name}_amass.npz')
    
    np.savez(out_path,
        poses=poses,
        betas=betas,
        trans=trans,
        gender=args.gender,
        mocap_framerate=args.fps
    )
    
    print(f"\nSaved: {out_path}")
    print(f"  poses:  {poses.shape} {poses.dtype}")
    print(f"  betas:  {betas.shape} {betas.dtype}")
    print(f"  trans:  {trans.shape} {trans.dtype}")
    print(f"  gender: {args.gender}")
    print(f"  fps:    {args.fps}")

if __name__ == '__main__':
    main()
