import argparse
from sfm_origin import SFM_ORIGIN
from sfm_origin_test import SFM_TEST
from utils import save_point_cloud_ply, remove_outliers_dbscan
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Structure-from-Motion Pipeline")
    parser.add_argument('--path', type=str, default="./data/", help="Directory containing images")
    parser.add_argument('--image_num', type=int, default=3, help="Number of images to use")
    parser.add_argument('--output_name', type=str, default="Moon_pst3D", help="Output PLY filename prefix")
    parser.add_argument('--mode', type=str, default='clean', choices=['raw','raw_clean', 'ba', 'clean'],
                        help='Pipeline mode: raw / ba / clean / raw_clean')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'],
                        help='Device for bundle adjustment')
    parser.add_argument('--sequence', action='store_true', help='Save PLY files for all views if enabled')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.path.endswith("data"):
        sfm = SFM_TEST()
    else:
        sfm = SFM_ORIGIN()
    
    apply_clustering = args.mode not in ['raw', 'raw_clean']
    
    sfm.set_dir_path(args.image_num, args.path)
    sfm.load_and_extract_features(apply_clustering=apply_clustering)

    if not sfm.map3d():
        print("Could not obtain 3D mapping")
        return
    
    pts3f_raw = [pt.pt for pt in sfm.recon3Dpts]
    
    if args.sequence:
        print(f"Generating progressive reconstruction PLY files (2 to {args.image_num} images)...")
        
        for i in range(2, args.image_num + 1):
            print(f"Processing reconstruction with {i} images")
            
            if args.path.endswith("data"):
                step_sfm = SFM_TEST()
            else:
                step_sfm = SFM_ORIGIN()
                
            step_sfm.set_dir_path(i, args.path)
            step_sfm.load_and_extract_features(apply_clustering=apply_clustering)
            
            if not step_sfm.map3d():
                print(f"Could not obtain 3D mapping for {i} images")
                continue
            
            step_pts3f_raw = [pt.pt for pt in step_sfm.recon3Dpts]
            
            step_ply_raw = f"{args.output_name}_{i}view_raw.ply"
            save_point_cloud_ply(step_ply_raw, step_sfm.get_dir_path(), 
                                step_pts3f_raw, step_sfm.rgb_values, 
                                step_sfm.cam3Dpts, step_sfm.cam3DRGB)
            
            if args.mode in ['ba', 'clean']:
                step_sfm.adjust_current_bundle(args.device)
                step_pts3f_ba = [pt.pt for pt in step_sfm.recon3Dpts]
                step_ply_ba = f"{args.output_name}_{i}view_ba.ply"
                save_point_cloud_ply(step_ply_ba, step_sfm.get_dir_path(), 
                                    step_pts3f_ba, step_sfm.rgb_values,
                                    step_sfm.cam3Dpts, step_sfm.cam3DRGB)
                
                if args.mode == 'clean':
                    step_sfm.get_cam_position()
                    step_pts_array = np.array(step_pts3f_ba)
                    step_cleaned_pts, step_inlier_mask = remove_outliers_dbscan(step_pts_array, eps=0.2, min_samples=10)
                    step_rgb_cleaned = [rgb for i, rgb in enumerate(step_sfm.rgb_values) if step_inlier_mask[i]]
                    step_ply_cleaned = f"{args.output_name}_{i}view_cleaned.ply"
                    save_point_cloud_ply(step_ply_cleaned, step_sfm.get_dir_path(), 
                                        step_cleaned_pts, step_rgb_cleaned,
                                        step_sfm.cam3Dpts, step_sfm.cam3DRGB)
            
            elif args.mode == 'raw_clean':
                step_sfm.get_cam_position()
                step_pts_array = np.array(step_pts3f_raw)
                step_cleaned_pts, step_inlier_mask = remove_outliers_dbscan(step_pts_array, eps=0.2, min_samples=10)
                step_rgb_cleaned = [rgb for i, rgb in enumerate(step_sfm.rgb_values) if step_inlier_mask[i]]
                step_ply_cleaned = f"{args.output_name}_{i}view_cleaned.ply"
                save_point_cloud_ply(step_ply_cleaned, step_sfm.get_dir_path(), 
                                    step_cleaned_pts, step_rgb_cleaned,
                                    step_sfm.cam3Dpts, step_sfm.cam3DRGB)
    
    ply_raw = f"{args.output_name}_{args.image_num}view_raw.ply"
    save_point_cloud_ply(ply_raw, sfm.get_dir_path(), pts3f_raw, sfm.rgb_values, sfm.cam3Dpts, sfm.cam3DRGB)
    print(f"Saved raw point cloud to {ply_raw}")
    
    pts3f_ba = pts3f_raw
    if args.mode in ['ba', 'clean']:
        print("Performing bundle adjustment...")
        sfm.adjust_current_bundle(args.device)
        pts3f_ba = [pt.pt for pt in sfm.recon3Dpts]
        ply_ba = f"{args.output_name}_{args.image_num}view_ba.ply"
        save_point_cloud_ply(ply_ba, sfm.get_dir_path(), pts3f_ba, sfm.rgb_values, sfm.cam3Dpts, sfm.cam3DRGB)
        print(f"Saved bundle-adjusted point cloud to {ply_ba}")

    if args.mode in ['clean', 'raw_clean']:
        print("Removing outliers with DBSCAN...")
        sfm.get_cam_position()
        target_pts = pts3f_ba if args.mode == 'clean' else pts3f_raw
        pts_array = np.array(target_pts)
        cleaned_pts, inlier_mask = remove_outliers_dbscan(pts_array, eps=0.2, min_samples=10)
        rgb_cleaned = [rgb for i, rgb in enumerate(sfm.rgb_values) if inlier_mask[i]]
        ply_cleaned = f"{args.output_name}_{args.image_num}view_cleaned.ply"
        save_point_cloud_ply(ply_cleaned, sfm.get_dir_path(), cleaned_pts, rgb_cleaned, sfm.cam3Dpts, sfm.cam3DRGB)
        print(f"Saved cleaned point cloud to {ply_cleaned}")

    print("Finished full SfM pipeline")

if __name__ == "__main__":
    main()