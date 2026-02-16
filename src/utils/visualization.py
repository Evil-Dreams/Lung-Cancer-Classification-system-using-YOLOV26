import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from skimage import measure

class LungVisualizer:
    """
    Class for visualizing lung CT scans, detections, and segmentations.
    Supports 2D slices, 3D volumes, and animations.
    """
    
    @staticmethod
    def display_slice(image: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     detections: Optional[Dict] = None,
                     figsize: Tuple[int, int] = (10, 10),
                     cmap: str = 'gray',
                     alpha: float = 0.3,
                     save_path: Optional[str] = None):
        """
        Display a single CT slice with optional mask and detections.
        
        Args:
            image (np.ndarray): 2D CT slice (H, W)
            mask (np.ndarray, optional): Binary mask (H, W)
            detections (Dict, optional): Dictionary of detections from YOLONoduleDetector
            figsize (tuple): Figure size (width, height)
            cmap (str): Colormap for the image
            alpha (float): Transparency for mask overlay
            save_path (str, optional): If provided, save the figure to this path
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Display the image
        ax.imshow(image, cmap=cmap)
        
        # Overlay mask if provided
        if mask is not None:
            # Create a colored mask
            mask_colored = np.zeros((*mask.shape, 4))
            mask_colored[mask > 0] = [1, 0, 0, alpha]  # Red with transparency
            ax.imshow(mask_colored)
        
        # Draw detections if provided
        if detections and 'boxes' in detections:
            for i in range(detections['count']):
                x1, y1, x2, y2 = detections['boxes'][i]
                score = detections['scores'][i]
                class_name = detections['class_names'][i]
                
                # Create a rectangle patch
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='lime', facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label = f"{class_name}: {score:.2f}"
                ax.text(
                    x1, y1 - 5, label,
                    color='white', fontsize=10,
                    bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none')
                )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def display_volume_slices(volume: np.ndarray,
                            masks: Optional[np.ndarray] = None,
                            detections: Optional[Dict[int, Dict]] = None,
                            n_slices: int = 5,
                            cmap: str = 'gray',
                            alpha: float = 0.3,
                            figsize: Tuple[int, int] = (15, 10)):
        """
        Display multiple slices from a 3D volume with optional masks and detections.
        
        Args:
            volume (np.ndarray): 3D volume (D, H, W)
            masks (np.ndarray, optional): 3D mask (D, H, W)
            detections (Dict[int, Dict], optional): Dictionary mapping slice indices to detections
            n_slices (int): Number of slices to display
            cmap (str): Colormap for the image
            alpha (float): Transparency for mask overlay
            figsize (tuple): Figure size (width, height)
        """
        # Select evenly spaced slices
        depth = volume.shape[0]
        step = max(1, depth // n_slices)
        slice_indices = range(0, depth, step)[:n_slices]
        
        # Create subplots
        n_cols = min(5, n_slices)
        n_rows = (n_slices + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.ravel()
        
        for i, idx in enumerate(slice_indices):
            if i >= len(axes):
                break
                
            ax = axes[i]
            ax.imshow(volume[idx], cmap=cmap)
            
            # Overlay mask if provided
            if masks is not None and idx < masks.shape[0]:
                mask_colored = np.zeros((*volume.shape[1:], 4))
                mask_colored[masks[idx] > 0] = [1, 0, 0, alpha]
                ax.imshow(mask_colored)
            
            # Draw detections if provided
            if detections and idx in detections:
                dets = detections[idx]
                for j in range(dets['count']):
                    x1, y1, x2, y2 = dets['boxes'][j]
                    score = dets['scores'][j]
                    class_name = dets['class_names'][j]
                    
                    # Create a rectangle patch
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=1, edgecolor='lime', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label for high confidence detections
                    if score > 0.5:  # Only show label for high confidence
                        ax.text(
                            x1, y1 - 5, f"{class_name}:{score:.2f}",
                            color='white', fontsize=8,
                            bbox=dict(facecolor='lime', alpha=0.7, edgecolor='none')
                        )
            
            ax.set_title(f"Slice {idx}")
            ax.axis('off')
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_volume_animation(volume: np.ndarray,
                              masks: Optional[np.ndarray] = None,
                              output_path: str = 'animation.gif',
                              fps: int = 10,
                              cmap: str = 'gray',
                              alpha: float = 0.3):
        """
        Create an animated GIF of a 3D volume.
        
        Args:
            volume (np.ndarray): 3D volume (D, H, W)
            masks (np.ndarray, optional): 3D mask (D, H, W)
            output_path (str): Output file path for the animation
            fps (int): Frames per second for the animation
            cmap (str): Colormap for the image
            alpha (float): Transparency for mask overlay
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(i):
            ax.clear()
            ax.imshow(volume[i], cmap=cmap)
            
            # Overlay mask if provided
            if masks is not None and i < masks.shape[0]:
                mask_colored = np.zeros((*volume.shape[1:], 4))
                mask_colored[masks[i] > 0] = [1, 0, 0, alpha]
                ax.imshow(mask_colored)
            
            ax.set_title(f"Slice {i}")
            ax.axis('off')
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=volume.shape[0],
            interval=1000//fps, blit=False
        )
        
        # Save animation
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        ani.save(output_path, writer='pillow', fps=fps)
        plt.close()
    
    @staticmethod
    def plot_3d_volume(volume: np.ndarray,
                      mask: Optional[np.ndarray] = None,
                      threshold: float = 0.5,
                      step_size: int = 2,
                      figsize: Tuple[int, int] = (10, 10)):
        """
        Create a 3D visualization of a volume with optional mask.
        
        Args:
            volume (np.ndarray): 3D volume (D, H, W)
            mask (np.ndarray, optional): 3D mask (D, H, W)
            threshold (float): Threshold for creating surface from mask
            step_size (int): Step size for downsampling the volume (for performance)
            figsize (tuple): Figure size (width, height)
        """
        # Downsample the volume for better performance
        if step_size > 1:
            volume = volume[::step_size, ::step_size, ::step_size]
            if mask is not None:
                mask = mask[::step_size, ::step_size, ::step_size]
        
        # Create a 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the volume surface
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5)
        mesh = Poly3DCollection(verts[faces], alpha=0.1, linewidth=0.5, edgecolor='k')
        mesh.set_facecolor([0.5, 0.5, 1])
        ax.add_collection3d(mesh)
        
        # Plot the mask surface if provided
        if mask is not None:
            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=threshold)
                mesh = Poly3DCollection(verts[faces], alpha=0.5, linewidth=0.5, edgecolor='r')
                mesh.set_facecolor([1, 0, 0])
                ax.add_collection3d(mesh)
            except:
                print("Could not create 3D surface from mask.")
        
        # Set the view
        ax.set_xlim(0, volume.shape[2])
        ax.set_ylim(0, volume.shape[1])
        ax.set_zlim(0, volume.shape[0])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_detection_histogram(detections: Dict[int, Dict],
                               class_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (10, 5)):
        """
        Plot a histogram of detections across slices.
        
        Args:
            detections (Dict[int, Dict]): Dictionary mapping slice indices to detections
            class_names (List[str], optional): List of class names
            figsize (tuple): Figure size (width, height)
        """
        if not detections:
            print("No detections to plot.")
            return
        
        # Count detections per class
        class_counts = {}
        for slice_idx, dets in detections.items():
            for i in range(dets['count']):
                class_id = dets['class_ids'][i]
                class_name = dets['class_names'][i] if class_names is None else class_names[class_id]
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
        
        # Create bar plot
        plt.figure(figsize=figsize)
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Number of Detections per Class')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Create distribution across slices
        slice_indices = list(detections.keys())
        slice_counts = [detections[idx]['count'] for idx in slice_indices]
        
        plt.figure(figsize=(12, 4))
        plt.bar(slice_indices, slice_counts)
        plt.title('Detections Across Slices')
        plt.xlabel('Slice Index')
        plt.ylabel('Number of Detections')
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example data
    volume = np.random.rand(50, 256, 256)  # Example 3D volume
    mask = (volume > 0.8).astype(np.float32)  # Example mask
    
    # Example detections
    detections = {
        10: {
            'boxes': [[50, 50, 100, 100], [150, 150, 200, 200]],
            'scores': [0.95, 0.87],
            'class_ids': [0, 0],
            'class_names': ['nodule', 'nodule'],
            'count': 2
        },
        20: {
            'boxes': [[80, 80, 130, 130]],
            'scores': [0.92],
            'class_ids': [0],
            'class_names': ['nodule'],
            'count': 1
        }
    }
    
    # Create visualizations
    visualizer = LungVisualizer()
    
    # Display a single slice
    visualizer.display_slice(
        volume[10], 
        mask[10] if mask is not None else None,
        detections.get(10) if detections else None
    )
    
    # Display multiple slices
    visualizer.display_volume_slices(
        volume, 
        mask,
        detections,
        n_slices=5
    )
    
    # Create 3D visualization
    visualizer.plot_3d_volume(
        volume,
        mask,
        step_size=2
    )
    
    # Plot detection histogram
    visualizer.plot_detection_histogram(
        detections,
        class_names=['nodule']
    )
    
    # Create animation (uncomment to use, requires saving to file)
    # visualizer.create_volume_animation(
    #     volume,
    #     mask,
    #     output_path='lung_animation.gif',
    #     fps=10
    # )
