Image Transformations for 2D Shapes
This project implements image transformations of translation, rotation, scaling, and shearing on a 2D shape along the X and Y axes. Each transformation requires a specific size of the transformation matrix:

    For translation of a 2D shape, a 3x3 transformation matrix is required.
    For other transformations (rotation, scaling, and shearing), a 2x2 matrix is sufficient.

By adjusting the sliders corresponding to each transformation, the elements in the transformation matrix change accordingly. These changes are observable in the shape's transformation, demonstrating how altering one or more elements in the matrix facilitates different types of transformations.

In the matrix given in the screen, following elements are affected for each transformation
	Translation:
		Row 1 Coulmn 3 is changed for movement along X axis (Top right of 3D matrix)
		Row 2 Coulmn 3 is changed for movement along Y axis (Second row, right most column of 3D matrix)
	Scaling:
		Upper left of 2D matrix for scaling along X axis. Make this negative to get a reflection along X axis
		Bottom right of 2D matrix for scaling along Y axis. Make this negative to get a reflection along Y axis
	Shearing:
		Upper right corresponds to horizondal shear (X-axis)
		Bottom left corresponds to vertical shear (Y-axis)
	Rotation:
		Rotation affects all the elements of the matrix. For a rotation of angle θ, matrix values are,
		cos(θ)    -sin(θ)
		sin(θ)     cos(θ)

For 3D shapes, an additional dimension in the matrix would be necessary to accommodate the extra axis of transformation.

Installation:
To run this tool, ensure you have numpy and matplotlib installed. You can install it using pip:
pip install numpy
pip install matplotlib