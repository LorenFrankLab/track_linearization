"""Demo script for interactive track builder.

This script demonstrates how to use make_track_from_image_interactive()
to build track graphs from images of experimental setups.
"""

import matplotlib.pyplot as plt
import numpy as np

from track_linearization import (
    get_linearized_position,
    make_track_from_image_interactive,
    plot_track_graph,
)


def create_demo_image():
    """Create a demo image with a simple maze drawn on it."""
    # Create a simple maze image
    img = np.ones((400, 600, 3))  # White background

    # Draw a T-maze in gray
    # Vertical stem
    img[100:300, 295:305, :] = 0.3

    # Horizontal top bar
    img[95:105, 150:450, :] = 0.3

    # Left arm
    img[100:250, 145:155, :] = 0.3

    # Right arm
    img[100:250, 445:455, :] = 0.3

    # Add some text
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.text(
        300,
        50,
        "Demo T-Maze",
        fontsize=20,
        ha="center",
        color="black",
        fontweight="bold",
    )
    ax.text(
        300,
        350,
        "Click to build track graph!",
        fontsize=14,
        ha="center",
        color="blue",
    )
    ax.axis("off")

    # Save demo image
    plt.savefig("/tmp/demo_maze.png", dpi=100, bbox_inches="tight")
    plt.close()

    return "/tmp/demo_maze.png"


def demo_basic_usage():
    """Demo 1: Basic usage with a demo image."""
    print("=" * 70)
    print("DEMO 1: Basic Interactive Track Builder")
    print("=" * 70)
    print("\nCreating a demo maze image...")

    image_path = create_demo_image()
    print(f"✓ Created demo image at {image_path}")

    print("\n" + "-" * 70)
    print("INSTRUCTIONS:")
    print("  1. Left-click on the maze to add nodes (place at track junctions)")
    print("  2. Right-click nodes to connect them with edges")
    print("  3. Press 'f' when finished")
    print("-" * 70)

    # Note: This would open an interactive window
    # For automated testing, we'll show the code but not run it
    print("\nTo run interactively:")
    print(">>> result = make_track_from_image_interactive(")
    print(f"...     image_path='{image_path}',")
    print("...     scale=0.1,  # 0.1 cm per pixel")
    print("... )")
    print(">>> track = result['track_graph']")
    print(">>> plot_track_graph(track)")


def demo_with_numpy_array():
    """Demo 2: Using numpy array instead of file path."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Using NumPy Array")
    print("=" * 70)

    # Create image array
    img = np.random.rand(300, 400, 3) * 0.2 + 0.8  # Light gray noise

    print("\nUsage with numpy array:")
    print(">>> import numpy as np")
    print(">>> img = np.random.rand(300, 400, 3)  # Your image")
    print(">>> result = make_track_from_image_interactive(")
    print("...     image_array=img,")
    print("...     scale=1.0,  # Keep pixel coordinates")
    print("... )")


def demo_workflow():
    """Demo 3: Complete workflow from image to linearization."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Complete Workflow")
    print("=" * 70)

    print("\nComplete workflow for extracting track from experimental photo:")
    print(
        """
# Step 1: Load your experimental setup image
result = make_track_from_image_interactive(
    image_path='path/to/your/maze_photo.jpg',
    scale=0.1,  # Adjust based on your image (cm per pixel)
)

# Step 2: Get the track graph
track = result['track_graph']

if track is not None:
    # Step 3: Visualize to verify
    from track_linearization import plot_track_graph
    plot_track_graph(track, show=True)

    # Step 4: Use for linearization
    import numpy as np
    from track_linearization import get_linearized_position

    # Your position data (in same units as track)
    positions = np.array([
        [10, 20],
        [15, 30],
        [20, 40],
    ])

    # Linearize
    result = get_linearized_position(positions, track)
    print(result)
"""
    )


def demo_keyboard_shortcuts():
    """Demo 4: Show all keyboard shortcuts and controls."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Keyboard Shortcuts & Controls")
    print("=" * 70)

    print(
        """
MOUSE CONTROLS:
  • Left Click:       Add a new node at cursor position
  • Right Click:      Select node (click twice on different nodes to create edge)
  • Middle Click:     Delete node (also removes all edges connected to it)

KEYBOARD SHORTCUTS:
  • 'f' key:          Finish and create the track graph
  • 'q' key:          Quit without creating a track
  • 'u' key:          Undo last node
  • 'd' key:          Delete last edge
  • 'h' key:          Show help/instructions

TIPS:
  • Node numbers are displayed above each node
  • Selected nodes are highlighted in yellow
  • Edges are drawn as blue lines
  • You can zoom/pan using matplotlib toolbar before adding nodes
  • Scale factor converts pixel coordinates to real-world units
"""
    )


def demo_tips_and_tricks():
    """Demo 5: Tips for best results."""
    print("\n\n" + "=" * 70)
    print("DEMO 5: Tips & Best Practices")
    print("=" * 70)

    print(
        """
1. CALIBRATION:
   • Measure a known distance in your image (e.g., track width)
   • Calculate scale = real_distance / pixel_distance
   • Example: If 50 pixels = 10 cm, scale = 10/50 = 0.2 cm/pixel

2. NODE PLACEMENT:
   • Place nodes at track junctions and endpoints
   • Use fewer nodes for straight sections (2 per segment)
   • Use more nodes for curved sections (smoother approximation)

3. EDGE ORDER:
   • The order you create edges doesn't matter
   • NetworkX will handle the graph structure
   • Use infer_edge_layout() later for automatic ordering

4. IMAGE QUALITY:
   • High contrast images work best
   • Clear boundaries make placement easier
   • Top-down photos are ideal (minimize perspective distortion)

5. JUPYTER NOTEBOOKS:
   • Use %matplotlib widget for best interactivity
   • Or use %matplotlib notebook (slightly less responsive)
   • Avoid %matplotlib inline (not interactive)

EXAMPLE CALIBRATION:
>>> # If you know the track width is 10 cm and measures 50 pixels:
>>> scale = 10.0 / 50.0  # = 0.2 cm per pixel
>>> result = make_track_from_image_interactive(
...     'maze.jpg',
...     scale=scale
... )
"""
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INTERACTIVE TRACK BUILDER - DEMONSTRATION")
    print("=" * 70)
    print("\nThis script shows how to use make_track_from_image_interactive()")
    print("to build track graphs from images of experimental setups.")

    # Run demos
    demo_basic_usage()
    demo_with_numpy_array()
    demo_workflow()
    demo_keyboard_shortcuts()
    demo_tips_and_tricks()

    print("\n\n" + "=" * 70)
    print("✓ Demo complete!")
    print("=" * 70)
    print("\nTo try it yourself:")
    print("  1. Load your image (maze photo, diagram, etc.)")
    print("  2. Call make_track_from_image_interactive()")
    print("  3. Click to build your track!")
    print("  4. Press 'f' when done")
    print("\nSee documentation for more details.")
    print("=" * 70)
