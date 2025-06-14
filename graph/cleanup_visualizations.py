import os
import glob

def remove_path_visualizations():
    """
    Removes all PNG files from path_visualizations subdirectories.
    """
    # Define the base directories
    base_dirs = ['visualize_high_risk', 'visualize_median', 'visualize_mean', 'visualize_overlap']
    
    # Counter for removed files
    removed_count = 0
    
    # Process each base directory
    for base_dir in base_dirs:
        base_dir = os.path.join(os.path.dirname(__file__), 'path_visualizations', base_dir)
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist, skipping...")
            continue
            
        # Find all PNG files in the directory
        png_files = glob.glob(os.path.join(base_dir, '*.png'))
        
        # Remove each PNG file
        for png_file in png_files:
            try:
                os.remove(png_file)
                removed_count += 1
                print(f"Removed: {png_file}")
            except Exception as e:
                print(f"Error removing {png_file}: {str(e)}")
    
    print(f"\nTotal files removed: {removed_count}")

if __name__ == "__main__":
    remove_path_visualizations() 